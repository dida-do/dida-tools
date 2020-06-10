"""
Contains wrapper for torch.utils.data.Dataset derived classes
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

DATA_FORMATS = {
    'npy': {
        'suffixes': ['npy'],
        'import_function': np.load,
        'to_npy_converter': lambda x: x},
    'tensor': {
        'suffixes': ['pt'],
        'import_function': torch.load,
        'to_npy_converter': lambda x: x.numpy()},
    'csv': {
        'suffixes': ['csv'],
        'import_function': pd.read_csv,
        'to_npy_converter': lambda x: x.to_numpy()},
    'excel': {
        'suffixes': ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf'],
        'import_function': pd.read_excel,
        'to_npy_converter': lambda x: x.to_numpy()},
    'hdf': {
        'suffixes': ['hdf', 'h5', 'hdf5', 'he5'],
        'import_function': pd.read_hdf,
        'to_npy_converter': lambda x: x.to_numpy()},
    'json': {
        'suffixes': ['json'],
        'import_function': pd.read_json,
        'to_npy_converter': lambda x: x.to_numpy()},
    'html': {
        'suffixes': ['html'],
        'import_function': pd.read_html,
        'to_npy_converter': lambda x: x.to_numpy()},
}


def check_for_data_format(suffix: str) -> str:
    '''
    Checks if a suffix corresponds to one of the DATA_FORMATS.

    Example:
    check_for_data_format('xlsx') -> 'excel'
    '''
    for data_format in DATA_FORMATS:
        if suffix in DATA_FORMATS[data_format]['suffixes']:
            return data_format
    return None


def check_for_file_with_supported_format(directory: str) -> str:
    '''
    Searches for a file with a supported format in a specified directory
    and returns the format. The format has to be specified in DATA_FORMATS.
    Raises an error if it does not find any supported file.

    Example:
    check_for_data_format('data/x/') -> 'csv'
    '''
    for item in os.listdir(directory):
        data_format = check_for_data_format(item.split('.')[-1])
        if data_format:
            return data_format

    raise FileNotFoundError('No files with supported suffixes found under DATA_DIR')


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

        self.x_format = check_for_file_with_supported_format(self.x_dir)
        self.y_format = check_for_file_with_supported_format(self.y_dir)

        # sort is needed for order in data
        self.x_list = np.sort(os.listdir(x_dir))
        self.y_list = np.sort(os.listdir(y_dir))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx: int) -> tuple:
        img_name = os.path.join(self.x_dir, self.x_list[idx])
        img = np.load(img_name)
        img = DATA_FORMATS[self.x_format]['import_function'](img_name)
        img = DATA_FORMATS[self.x_format]['to_npy_converter'](img)

        label_name = os.path.join(self.y_dir, self.y_list[idx])
        label = DATA_FORMATS[self.y_format]['import_function'](label_name)
        label = DATA_FORMATS[self.y_format]['to_npy_converter'](label)

        label = (label > 0).astype(float)

        img_tensor = torch.Tensor(img)
        label_tensor = torch.Tensor(label)
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


@dataclass
class InpaintingDataset(torch.utils.data.Dataset):
    """Dataset for inpainting from numpy files

    :param root_dir: Directory containing x and y directories
    :param aug: data Augmentation pipeline.
    :param input_preprocess: Additional deterministic preprocessing for input
    :param min_val: The minimum value to clip input data.
    :param max_val: The maximum value to clip input data.
    :param fnames: Optional subset of files to use.
    :param mask_fn: Function to create masks.
    """

    root_dir: Path
    aug: Callable
    input_preprocess: Optional[Callable]
    min_val: int = 0
    max_val: int = 6000
    fnames: Optional[List[str]] = None
    mask_fn: Callable = lambda x: np.zeros_like(x)

    def __post_init__(self):
        super().__init__()

        if self.fnames is None:
            self.fnames = [f for f in os.listdir(self.root_dir / "x") if f.endswith(".npy")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        x = np.clip((np.load(self.root_dir / "x" / self.fnames[idx]) + self.min_val) / self.max_val, 0, 1)
        x = (x * 255).astype(np.uint8).transpose(1, 2, 0)

        if self.aug is not None:
            initial_shape = x.shape
            x = self.aug(image=x)["image"]

            # Some augmentations are not suited to multi-channel data and silently change it to 3 channels
            assert x.shape == initial_shape

        mask = self.mask_fn(x).transpose(2, 0, 1).astype(np.float16)

        if self.input_preprocess is not None:
            x = self.input_preprocess(x)
        else:
            x = torch.Tensor(x).transpose(2, 0, 1)

        return x, torch.Tensor(mask).float()


@dataclass
class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for segmentation from numpy files

    :param root_dir: Directory containing x and y directories
    :param aug: data Augmentation pipeline.
    :param input_preprocess: Additional deterministic preprocessing for input
    :param min_val: The minimum value to clip input data.
    :param max_val: The maximum value to clip input data.
    :param fnames: Optional subset of files to use.
    """

    root_dir: Path
    aug: Callable
    input_preprocess: Optional[Callable]
    min_val: int = 0
    max_val: int = 6000
    fnames: Optional[List[str]] = None

    def __post_init__(self):
        super().__init__()

        if self.fnames is None:
            self.fnames = [f for f in os.listdir(self.root_dir / "x") if f.endswith(".npy")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        x = np.clip((np.load(self.root_dir / "x" / self.fnames[idx]) + self.min_val) / self.max_val, 0, 1)
        x = (x * 255).astype(np.uint8).transpose(1, 2, 0)

        y = np.load(self.root_dir / "y" / self.fnames[idx])
        if len(y.shape) == 3:
            if (x.shape == y.shape) or (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]):
                # channel order the same
                pass
            elif (x.shape[0] == y.shape[1] and x.shape[1] == y.shape[2]):
                # x channels last, y channels first
                y = y.transpose(1, 2, 0)
            elif (x.shape[1] == y.shape[0] and x.shape[2] == y.shape[1]):
                # x channels first, y channels last
                x = x.transpose(1, 2, 0)
            else:
                raise ValueError(f"x dimensions {x.shape} not compatible with y dimiensions {y.shape}")

        elif len(y.shape) == 2:
            y = y[:, :, None]
        else:
            raise ValueError(f"Target {self.fnames[idx]} has invalid dimensions")

        if self.aug is not None:
            initial_shape = x.shape
            initial_y_shape = y.shape
            augmented = self.aug(image=x, mask=y)

            x = augmented["image"]
            y = augmented["mask"]

            # Some augmentations are not suited to multi-channel data and silently change it to 3 channels

            assert x.shape == initial_shape
            assert y.shape == initial_y_shape

        if self.input_preprocess is not None:
            x = self.input_preprocess(x)
        else:
            x = torch.Tensor(x).transpose(2, 0, 1)

        return x, torch.Tensor(y).float().permute(2, 0, 1)


@dataclass
class UnpairedImageDataset(torch.utils.data.Dataset):
    """Dataset for two sets of unpaired images eg for a cycleGAN.

    :param root_dir: Directory containing x and y directories

    All following params come in x and y flavours.

    :param aug: Data augmentation pipeline.
    :param input_preprocess: Additional deterministic preprocessing for input
    :param min_val: The minimum value to clip input data.
    :param max_val: The maximum value to clip input data.
    :param fnames: Optional subset of files to use."""

    root_dir: Path
    x_aug: Callable
    x_input_preprocess: Optional[Callable]

    y_aug: Callable
    y_input_preprocess: Optional[Callable]

    x_min_val: int = 0
    x_max_val: int = 6000
    x_fnames: Optional[List[str]] = None

    y_min_val: int = 0
    y_max_val: int = 6000
    y_fnames: Optional[List[str]] = None

    def __post_init__(self):
        super().__init__()

        if self.x_fnames is None:
            self.x_fnames = [f for f in os.listdir(self.root_dir / "x") if f.endswith(".npy")]

        if self.y_fnames is None:
            self.y_fnames = [f for f in os.listdir(self.root_dir / "y") if f.endswith(".npy")]

    def __len__(self):
        return len(self.x_fnames)

    def __getitem__(self, idx):
        x = np.clip((np.load(self.root_dir / "x" / self.x_fnames[idx]) + self.x_min_val) / self.x_max_val, 0, 1)
        x = (x * 255).astype(np.uint8).transpose(1, 2, 0)

        y = np.clip((np.load(self.root_dir / "y" / self.y_fnames[idx]) + self.y_min_val) / self.y_max_val, 0, 1)
        y = (y * 255).astype(np.uint8).transpose(1, 2, 0)

        if self.x_aug is not None:
            initial_shape = x.shape
            x = self.x_aug(image=x)["image"]

            # Some augmentations are not suited to multi-channel data and silently change it to 3 channels
            assert x.shape == initial_shape

        if self.y_aug is not None:
            initial_shape = y.shape
            y = self.y_aug(image=y)["image"]

            # Some augmentations are not suited to multi-channel data and silently change it to 3
            assert y.shape == initial_shape, f"{y.shape} != {initial_shape}"
            # Tempoaray hack for 1 channel sentinel 1 data
            # Will be removed once data preparation has been fixed
            if y.shape[-1] == 1:
                y = np.stack(2 * [y[:, :, 0]], axis=-1)

        if self.x_input_preprocess is not None:
            x = self.x_input_preprocess(x)
        else:
            x = torch.Tensor(x).transpose(2, 0, 1)

        if self.y_input_preprocess is not None:
            y = self.y_input_preprocess(y)
        else:
            y = torch.Tensor(y).transpose(2, 0, 1)

        return x, y


class ImageBuffer:
    """Image buffer for training discriminator in a GAN after github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""

    def __init__(self, pool_size: int, replace_prob: float = 0.5):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []
        self.replace_prob = replace_prob

    def __call__(self, images: torch.Tensor):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) < self.replace_prob:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)
