"""
Contains wrapper for torch.utils.data.Dataset derived classes
"""

import os

import numpy as np
import pandas as pd
import torch.utils.data

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
