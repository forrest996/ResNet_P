import sys
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch
from torch.utils.data import DataLoader
import time
from argparse import ArgumentParser
from train_loader import CIFAR_100


from models.WideResNet_P2_tiny import wideresnet58_p2_4_tiny
model = wideresnet58_p2_4_tiny()


config = {
    'data_file': './data/CIFAR-100/',
}


@torch.no_grad()
def measure_latency(images, model, GPU=True, chan_last=False, half=False, num_threads=None, iter=200):
    """
    :param images: b, c, h, w
    :param model: model
    :param GPU: whther use GPU
    :param chan_last: data_format
    :param half: half precision
    :param num_threads: for cpu
    :return:
    """

    if GPU:
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True

        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()

        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(iter):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on gpu {throughput}")
        print(f"batch_size {batch_size} latency on gpu {latency} ms")

        return throughput, latency
    else:
        model.eval()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        batch_size = images.shape[0]

        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()
        for i in range(10):
            model(images)
        tic1 = time.time()
        for i in range(iter):
            model(images)
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on cpu {throughput}")
        print(f"batch_size {batch_size} latency on cpu {latency} ms")

        return throughput, latency


if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('-s', '--size', type=int, default='32', help='the size of image')
    args = parse.parse_args()

    test_data = CIFAR_100(data_path=config['data_file'], resize=args.size, model_selection='test',
                          use_pretreatment=True,
                          valid_size=0)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=True, num_workers=4, pin_memory=False)

    model = model.eval()

    print("start calc!")
    for images, label in test_loader:
        throughput, latency = measure_latency(images[:1, :, :, :], model, GPU=False, num_threads=1)
        if torch.cuda.is_available():
            throughput, latency = measure_latency(images, model, GPU=True)
        exit()
    print(model)






