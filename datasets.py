import os
import random
import h5py, cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from glob import glob


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class MedicalImage_dataset(Dataset):
    def __init__(self, base_dir, suffix_name, transform=None):
        self.transform = transform  # using transform in torch!
        self.suffix_name = suffix_name
        self.base_dir = base_dir
        self.data_name_list = glob(base_dir + "*" + self.suffix_name)

        self.data_dir = base_dir

    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx):
        if self.suffix_name == ".npz":
            data_name = self.data_name_list[idx]
            data = np.load(data_name)
            image, label = data['image'], data['label']

        elif self.suffix_name == ".npy.h5":
            data_name = self.data_name_list[idx]
            data = h5py.File(data_name)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.data_name_list[idx].replace(self.suffix_name, "").replace(self.base_dir, "")
        return sample


class MedicalImage_dataset_mixup(Dataset):
    def __init__(self, base_dir, suffix_name, transform=None):
        self.transform = transform  # using transform in torch!
        self.suffix_name = suffix_name
        self.base_dir = base_dir
        self.data_name_list = glob(base_dir + "*" + self.suffix_name)

        self.data_dir = base_dir

    def __len__(self):

        return len(self.data_name_list)

    def __getitem__(self, idx):
        image_mixup1, label_mixup1 = None, None
        image_mixup2, label_mixup2 = None, None

        if self.suffix_name == ".npz":
            data_name = self.data_name_list[idx]
            data = np.load(data_name)
            image, label = data['image'], data['label']

            idx2 = np.random.randint(0, self.__len__())
            data_name2 = self.data_name_list[idx2]
            data2 = np.load(data_name2)
            image2, label2 = data2['image'], data['label']

            img_shape = image.shape
            mask = np.ones(img_shape)
            mask[:, :img_shape[1] // 2] = 0
            image_mixup1 = mask * image + (1 - mask) * image2
            image_mixup2 = (1 - mask) * image + mask * image2
            label_mixup1 = mask * label + (1 - mask) * label2
            label_mixup2 = (1 - mask) * label + mask * label2


        elif self.suffix_name == ".npy.h5":
            data_name = self.data_name_list[idx]
            data = h5py.File(data_name)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}

        if image_mixup1 is None or image_mixup2 is None:
            sample['image_mixup'] = image
            sample['image_mixup2'] = image
        else:
            sample['image_mixup'] = image_mixup1
            sample['image_mixup2'] = image_mixup2
        if label_mixup1 is None or label_mixup2 is None:
            sample['label_mixup'] = label
            sample['label_mixup2'] = label
        else:
            sample["label_mixup"] = label_mixup1
            sample["label_mixup2"] = label_mixup2

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.data_name_list[idx].replace(self.suffix_name, "").replace(self.base_dir, "")
        return sample
