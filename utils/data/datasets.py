"""
Contains wrapper for torch.utils.data.Dataset derived classes
"""

import os
import random
import numpy as np
from torchvision.transforms import RandomApply, RandomChoice
import torch
import torch.utils.data
import utils.data.augmentation as aug

class NpyDataset(torch.utils.data.Dataset):
    '''
    A supervised learning dataset class to handle serialised
    numpy data, for example images.

    Data consists of float `.npy` files of fixed shape.
    Observations and labels are given by different folders
    containing files with same names.
    '''
    def __init__(self, x_dir, y_dir):
        """
        Instantiate .npy file dataset.

        :param x_dir: (str) observation directory
        :param y_dir: (str) label directory
        """

        self.x_dir = x_dir
        self.y_dir = y_dir

        # sort is needed for order in data
        self.x_list = np.sort(os.listdir(x_dir))
        self.y_list = np.sort(os.listdir(y_dir))

        img_transforms_list = [
            aug.HorizontalFlip(),
            aug.VerticalFlip(),
            aug.GaussianNoise(mean=0.5, std=0.05),
            aug.Rotate90left(),
            aug.Rotate90right(),
            aug.Rotate180()
        ]

        label_transforms_list = [
            aug.HorizontalFlip(),
            aug.VerticalFlip(),
            aug.Identity(),
            aug.Rotate90left(),
            aug.Rotate90right(),
            aug.Rotate180()
        ]

        self.img_transforms = RandomApply(RandomChoice(img_transforms_list), p=1/7)
        self.label_transforms = RandomApply(RandomChoice(label_transforms_list), p=1/7)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx: int) -> tuple:

        img_name = os.path.join(self.x_dir, self.x_list[idx])
        img = np.load(img_name)

        label_name = os.path.join(self.y_dir, self.y_list[idx])
        label = np.load(label_name)

        label = (label > 0).astype(float)

        img_tensor = torch.Tensor(img)
        label_tensor = torch.Tensor(label)

        seed1 = np.random.randint(2**32 - 1) # TODO should be sys.maxint for python2
        seed2 = np.random.randint(2**32 - 1) # TODO should be sys.maxint for python2

        random.seed(seed1)
        torch.manual_seed(seed2)
        img_tensor = self.img_transforms(img_tensor)

        random.seed(seed1)
        torch.manual_seed(seed2)
        label_tensor = self.label_transforms(label_tensor)

        return img_tensor, label_tensor


class NpyPredictionDataset(torch.utils.data.Dataset):
    '''
    A dataset class to handle prediction on serialised numpy data,
    for example images.

    Data consists of float `.npy` files of fixed shape.
    '''
    def __init__(self, files):
        """
        Instantiate .npy file dataset.

        :param files: (list) list of files to predict on
        """

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:

        file = np.load(self.files[idx])
        file = torch.Tensor(file)
        return self.files[idx], file
