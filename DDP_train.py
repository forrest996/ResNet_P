import sys
from os.path import dirname,abspath

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import argparse
import yaml
from argparse import Namespace

from tools.train_loader import CIFAR_100, CIFAR_10
from tools import utils
from models.WideResNet_P2_tiny import wideresnet58_p2_4_tiny


def extract_number(s):
    return int(s.split('_')[1])


def increment_out_path(file='./output'):
    out = ''
    path_list = os.listdir(file)
    path_list = [x for x in path_list if x != '.ipynb_checkpoints']
    path_list = sorted(path_list, key=extract_number)

    if not os.path.exists(os.path.join(file, 'exp_0')):
        os.makedirs(os.path.join(file, 'exp_0'), mode=0o755)
        out = os.path.join(file, 'exp_0')
    file_test_label = False
    for i in range(len(path_list)):
        for_test = 0
        matches = re.findall(r"(.*?)(\d+)", str(path_list[i]))
        stem, num = matches[-1]
        num = int(num)
        if num == for_test:
            file_test_label = True
            for_test += 1
        else:
            break

    if file_test_label == True:
        final_file_inx = len(path_list) - 1
        final_file = path_list[final_file_inx]
        matches = re.findall(r"(.*?)(\d+)", str(final_file))
        stem, num = matches[-1]
        out = os.path.join(file, f"{stem}{int(num) + 1}")
        out = out.replace('\\', '/')
        os.makedirs(out)
    print('running out file: ',out)

    return out


# lr
def warm_up_strategy_resnet(epoch, args):
    warm_up_epochs = args.warmup_epochs
    if epoch < warm_up_epochs:
        lr_lamda = (((1. - (args.lr_start/100.)) / warm_up_epochs) * (epoch + 1))
    elif epoch >= warm_up_epochs and epoch < int(args.epochs*0.5):
        lr_lamda = 1.
    elif epoch >= int(args.epochs*0.5) and epoch < int(args.epochs*0.75):
        lr_lamda = 0.1
    else:
        lr_lamda = 0.01

    return lr_lamda


def warm_up_strategy_wideresnet(epoch, args):
    warm_up_epochs = args.warmup_epochs
    if epoch < warm_up_epochs:
        lr_lamda = (((1. - (args.lr_start/100.)) / warm_up_epochs) * (epoch + 1))
    elif epoch >= warm_up_epochs and epoch < 60:
        lr_lamda = 1.
    elif epoch >= 60 and epoch < 120:
        lr_lamda = 0.2
    elif epoch >= 120 and epoch < 160:
        lr_lamda = 0.2 * 0.2
    else:
        lr_lamda = 0.2 * 0.2 * 0.2

    return lr_lamda


# setup process group. rank is GPU id, world_size is GPU nums
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# source：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


# get dataloader
def getloader(rank, world_size, args, pin_memory=True, num_workers=0):
    if 'CIFAR-100' in args.data_file:
        loader_data = CIFAR_100
    elif 'CIFAR-10' in args.data_file:
        loader_data = CIFAR_10
    train_data = loader_data(data_path=args.data_file, resize=None, model_selection='train',
                           use_pretreatment=True, valid_size=0)
    test_data = loader_data(data_path=args.data_file, resize=None, model_selection='test', use_pretreatment=True,
                          valid_size=0)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=True, shuffle=False, sampler=train_sampler)

    test_sampler = SequentialDistributedSampler(dataset=test_data, batch_size=args.test_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=test_sampler)
    return train_dataloader, test_dataloader


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# clean process
def cleanup():
    dist.destroy_process_group()


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def is_master_proc(num_gpus):
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)


def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)


# warp the model to DDP, main is running in every parallel process.
def main(rank, args, resume, out_path):
    seed = args.seed
    # lr strategy
    if args.lr_strategy == 0:
        warm_up_strategy = warm_up_strategy_resnet
    else:
        warm_up_strategy = warm_up_strategy_wideresnet

    # set seed, different process using dif seed.Avoid homomorphism issues.
    init_seeds(seed + rank)
    # set up the process groups
    setup(rank, args.world_size)

    # prepare the dataloader
    train_dataloader, test_dataloader = getloader(rank, args.world_size, args)

    # instantiate the model(it's your own model) and move it to the right device
    model = wideresnet58_p2_4_tiny().to(rank)

    # using SyncBN (don't use this in small dataset)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : warm_up_strategy(epoch, args))
    loss_ = nn.CrossEntropyLoss().to(rank)

    # resume
    if resume is not None:
        print('start resume training!')
        checkpoint = torch.load(resume, map_location="cpu")
        prefix_weights = {}
        for key, value in checkpoint["model"].items():
            prefix_weights['module.' + key] = value
        model.load_state_dict(prefix_weights)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
    # wait
    torch.distributed.barrier()

    # training
    best_accuracy, best_epoch = 0, 0
    xx_epoch, yy_accuracy = [], []
    for epoch in range(args.start_epoch, args.epochs):
        train_dataloader.sampler.set_epoch(epoch) # during use DistributedSampler
        model.train()
        for idx, (img, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, disable=not is_master_proc(num_gpus=args.world_size)):
            if torch.cuda.is_available():
                img = img.float().to(rank)
                label = label.to(rank)
            optimizer.zero_grad()
            predict = model(img)
            loss = loss_(predict, label)
            loss.backward()
            optimizer.step()
        # evaluate
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for data, label in test_dataloader:
                data, label = data.to(rank), label.to(rank)
                predictions.append(model(data))
                labels.append(label)

            predictions = distributed_concat(torch.concat(predictions, dim=0),
                                             len(test_dataloader.sampler.dataset))
            labels = distributed_concat(torch.concat(labels, dim=0),
                                        len(test_dataloader.sampler.dataset))
        # wait
        torch.distributed.barrier()
        # calc Top1, Top5, save on test top1 best model
        acc1, acc5 = utils.accuracy(predictions, labels, topk=(1, 5))
        yy_accuracy.append(acc1)
        # save checkpoint
        if rank == 0:
            print(f"epoch:{epoch}, lr:{scheduler.get_last_lr()[0]},test Acc@1:{acc1.item():.3f}, test Acc@5:{acc5.item():.3f}")
            if acc1 > best_accuracy:
                best_accuracy = acc1
                best_epoch = epoch
                torch.save(model.module.state_dict(), out_path + '/{}_best_seed{}.mdl'.format(args.model_save_name, seed))
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if epoch % 40 == 0:
                utils.save_on_master(checkpoint, os.path.join(out_path, f"model_{epoch}_seed{seed}.pth"))
            utils.save_on_master(checkpoint, os.path.join(out_path, "checkpoint.pth"))
        scheduler.step()
    if rank == 0 :
        for i in range(len(yy_accuracy)):
            yy_accuracy[i] = (100 - yy_accuracy[i].detach().cpu().numpy())
        print('model: {} ,best_epoch: {}, best_error: {}'.format(args.model_save_name, best_epoch, best_accuracy))
        plt.figure()
        plt.title('model: {} best_error: {}'.format(args.model_save_name, best_accuracy))
        plt.plot(np.arange(0, len(yy_accuracy), 1), yy_accuracy)
        plt.xlabel('epoch')
        plt.ylabel('test error (%)')
        plt.savefig(out_path + '/{}_test_error_seed{}'.format(args.model_save_name, seed))
        plt.show()
        if not os.path.exists(os.path.join(out_path, 'test_error_{}_seed{}'.format(args.model_save_name, seed) + '.txt')):
            with open(os.path.join(out_path, 'test_error_{}_seed{}'.format(args.model_save_name, seed) + '.txt'), 'w') as f:
                for top1 in yy_accuracy:
                    f.write(str(top1) + '\n')
                f.close()
    # wait
    torch.distributed.barrier()
    # clean
    cleanup()


if __name__ == '__main__':

    out_path = increment_out_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, type=str, help="path of checkpoint")
    parser.add_argument('-c', '--cfg', type=str, default='cfg/wideresnet.yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)
    print('args:\n', args)

    mp.spawn(main, args=(args , args.resume, out_path), nprocs=args.world_size)

