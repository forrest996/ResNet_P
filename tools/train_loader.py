import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import random

# CIFER-10
'''
        {b'batch_label': b'training batch 1 of 5', 
         b'labels': [6, 9 ... 1, 5], 
         b'data': array([[ 59,  43,  50, ..., 140,  84,  72],
                                         ...
                         [ 62,  61,  60, ..., 130, 130, 131]], dtype=uint8),
         b'filenames': [b'leptodactylus_pentadactylus_s_000004.png', b'camion_s_000148.png',
                                         ...
                        b'estate_car_s_001433.png', b'cur_s_000170.png']}
        '''


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Data transforms
mean=[0.49139968,  0.48215841,  0.44653091]
stdv= [0.24703223,  0.24348513,  0.26158784]

def get_image_transform(resize):
    if resize is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=mean, std=stdv)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])
        return transform_train, transform_test
    else:
        transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=mean, std=stdv)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])
        return transform_train, transform_test

class CIFER_10(Dataset):
    def __init__(self, data_path, resize=None, model_selection='train', use_pretreatment=True, valid_size=5000):
        super(CIFER_10, self).__init__()
        self.data_path = data_path
        self.resize = resize
        self.model_selection = model_selection
        self.use_pretreatment = use_pretreatment

        self.transform_train,  self.transform_test = get_image_transform(self.resize)

        self.data_buffer = []
        self.file = data_path
        path_list = os.listdir(os.path.join(self.file))
        # get train data
        for data_bach in path_list:
            if 'data' in data_bach:
                dict = unpickle(os.path.join(self.file, data_bach))
                for idx in range(len(dict[b'labels'])):
                    img_one = {'img': None, 'label': None}
                    img_one['img'] = dict[b'data'][idx].reshape(3, 32, 32) # (C, H, W)
                    img_one['label'] = dict[b'labels'][idx]
                    self.data_buffer.append(img_one)
        # split train and val data
        if valid_size > 0:
            indices = torch.randperm(len(self.data_buffer))
            train_indices = indices[:len(indices) - valid_size]
            valid_indices = indices[len(indices) - valid_size:]
            if model_selection ==  'train':
                self.data_buffer = torch.utils.data.Subset(self.data_buffer, train_indices)
            elif model_selection == 'val':
                self.data_buffer = torch.utils.data.Subset(self.data_buffer, valid_indices)
        if model_selection == 'test':
            self.data_buffer = []
            for data_bach in path_list:
                if 'test' in data_bach:
                    dict = unpickle(os.path.join(self.file, data_bach))
                    for idx in range(len(dict[b'labels'])):
                        img_one = {'img': None, 'label': None}
                        img_one['img'] = dict[b'data'][idx].reshape(3, 32, 32) # (C, H, W)
                        img_one['label'] = dict[b'labels'][idx]
                        self.data_buffer.append(img_one)
    def __len__(self):
        return len(self.data_buffer)
    def __getitem__(self, idx):
            if self.use_pretreatment == True:
                if self.model_selection == 'train':
                    return self.transform_train(Image.fromarray(self.data_buffer[idx]['img'].transpose(2, 1, 0).transpose(1, 0, 2))), torch.tensor(self.data_buffer[idx]['label'])
                elif self.model_selection == 'val' or self.model_selection == 'test':
                    return self.transform_test(Image.fromarray(self.data_buffer[idx]['img'].transpose(2, 1, 0).transpose(1, 0, 2))), torch.tensor(self.data_buffer[idx]['label'])
            else:
                return torch.tensor(self.data_buffer[idx]['img']), torch.tensor(self.data_buffer[idx]['label'])


class CIFER_100(Dataset):
    def __init__(self, data_path, resize=None, model_selection='train', use_pretreatment=True, valid_size=5000):
        super(CIFER_100, self).__init__()
        self.resize = resize
        self.model_selection = model_selection
        self.use_pretreatment = use_pretreatment
        self.transform_train,  self.transform_test = get_image_transform(self.resize)
        self.data_buffer = []
        self.file = data_path
        path_list = os.listdir(os.path.join(self.file))
        # get train data
        for data_bach in path_list:
            if 'train' in data_bach:
                dict = unpickle(os.path.join(self.file, data_bach))
                for idx in range(len(dict[b'fine_labels'])):
                    img_one = {'img': None, 'label': None}
                    img_one['img'] = dict[b'data'][idx].reshape(3, 32, 32) # (C, H, W)
                    img_one['label'] = dict[b'fine_labels'][idx]
                    self.data_buffer.append(img_one)
        # split train and val data
        if valid_size > 0:
            indices = torch.randperm(len(self.data_buffer))
            train_indices = indices[:len(indices) - valid_size]
            valid_indices = indices[len(indices) - valid_size:]
            if model_selection ==  'train':
                self.data_buffer = torch.utils.data.Subset(self.data_buffer, train_indices)
            elif model_selection == 'val':
                self.data_buffer = torch.utils.data.Subset(self.data_buffer, valid_indices)
        if model_selection == 'test':
            self.data_buffer = []
            for data_bach in path_list:
                if 'test' in data_bach:
                    dict = unpickle(os.path.join(self.file, data_bach))
                    for idx in range(len(dict[b'fine_labels'])):
                        img_one = {'img': None, 'label': None}
                        img_one['img'] = dict[b'data'][idx].reshape(3, 32, 32) # (C, H, W)
                        img_one['label'] = dict[b'fine_labels'][idx]
                        self.data_buffer.append(img_one)
    def __len__(self):
        return len(self.data_buffer)
    def __getitem__(self, idx):
            if self.use_pretreatment == True:
                if self.model_selection == 'train':
                    return self.transform_train(Image.fromarray(self.data_buffer[idx]['img'].transpose(2, 1, 0).transpose(1, 0, 2))), torch.tensor(self.data_buffer[idx]['label'])
                elif self.model_selection == 'val' or self.model_selection == 'test':
                    return self.transform_test(Image.fromarray(self.data_buffer[idx]['img'].transpose(2, 1, 0).transpose(1, 0, 2))), torch.tensor(self.data_buffer[idx]['label'])
            else:
                return torch.tensor(self.data_buffer[idx]['img']), torch.tensor(self.data_buffer[idx]['label'])



if __name__ == '__main__':
    '''
    test
    '''
    file = '../data/CIFER-100/'
    def set_seed(seed=None):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(1)
    data = CIFER_100(data_path=file, resize=None, model_selection='train', use_pretreatment=True, valid_size=0)
    print(data.__len__())
    classes = {19: '11-large_omnivores_and_herbivores', 29: '15-reptiles', 0: '4-fruit_and_vegetables', 11: '14-people',
     1: '1-fish', 86: '5-household_electrical_devices', 90: '18-vehicles_1', 28: '3-food_containers',
     23: '10-large_natural_outdoor_scenes', 31: '11-large_omnivores_and_herbivores',
     39: '5-household_electrical_devices', 96: '17-trees', 82: '2-flowers', 17: '9-large_man-made_outdoor_things',
     71: '10-large_natural_outdoor_scenes', 8: '18-vehicles_1', 97: '8-large_carnivores', 80: '16-small_mammals',
     74: '16-small_mammals', 59: '17-trees', 70: '2-flowers', 87: '5-household_electrical_devices',
     84: '6-household_furniture', 64: '12-medium_mammals', 52: '17-trees', 42: '8-large_carnivores', 47: '17-trees',
     65: '16-small_mammals', 21: '11-large_omnivores_and_herbivores', 22: '5-household_electrical_devices',
     81: '19-vehicles_2', 24: '7-insects', 78: '15-reptiles', 45: '13-non-insect_invertebrates',
     49: '10-large_natural_outdoor_scenes', 56: '17-trees', 76: '9-large_man-made_outdoor_things', 89: '19-vehicles_2',
     73: '1-fish', 14: '7-insects', 9: '3-food_containers', 6: '7-insects', 20: '6-household_furniture',
     98: '14-people', 36: '16-small_mammals', 55: '0-aquatic_mammals', 72: '0-aquatic_mammals',
     43: '8-large_carnivores', 51: '4-fruit_and_vegetables', 35: '14-people', 83: '4-fruit_and_vegetables',
     33: '10-large_natural_outdoor_scenes', 27: '15-reptiles', 53: '4-fruit_and_vegetables', 92: '2-flowers',
     50: '16-small_mammals', 15: '11-large_omnivores_and_herbivores', 18: '7-insects', 46: '14-people',
     75: '12-medium_mammals', 38: '11-large_omnivores_and_herbivores', 66: '12-medium_mammals',
     77: '13-non-insect_invertebrates', 69: '19-vehicles_2', 95: '0-aquatic_mammals', 99: '13-non-insect_invertebrates',
     93: '15-reptiles', 4: '0-aquatic_mammals', 61: '3-food_containers', 94: '6-household_furniture',
     68: '9-large_man-made_outdoor_things', 34: '12-medium_mammals', 32: '1-fish', 88: '8-large_carnivores',
     67: '1-fish', 30: '0-aquatic_mammals', 62: '2-flowers', 63: '12-medium_mammals',
     40: '5-household_electrical_devices', 26: '13-non-insect_invertebrates', 48: '18-vehicles_1',
     79: '13-non-insect_invertebrates', 85: '19-vehicles_2', 54: '2-flowers', 44: '15-reptiles', 7: '7-insects',
     12: '9-large_man-made_outdoor_things', 2: '14-people', 41: '19-vehicles_2', 37: '9-large_man-made_outdoor_things',
     13: '18-vehicles_1', 25: '6-household_furniture', 10: '3-food_containers', 57: '4-fruit_and_vegetables',
     5: '6-household_furniture', 60: '10-large_natural_outdoor_scenes', 91: '1-fish', 3: '8-large_carnivores',
     58: '18-vehicles_1', 16: '3-food_containers'}
    for idx in range(1):
        img, label = data.__getitem__(idx)
        print(label)
        print(img)
        plt.figure()
        plt.title(classes[int(label.detach().cpu())])
        print(type(img))
        print(img.shape)
        plt.imshow(img.transpose(0, 2).transpose(0, 1))
        plt.show()



