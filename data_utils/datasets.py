"""
Contains wrapper for torch.utils.data.Dataset derived classes
"""

import numpy as np
import torch

class NpyDataset(torch.utils.data.Dataset):
    '''
    A dataset class to handle serialised numpy data, for example images.

    Data consists of float `.npy` files of fixed shape.
    '''
    def __init__(self, x_dir, y_dir, tfms: Optional[callable]=None):
        self.x_dir = x_dir
        self.y_dir = y_dir

        self.x_list = os.listdir(x_dir)
        self.y_list = os.listdir(y_dir)

        self.tfms = tfms

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx: int) -> tuple:
        
        img_name = os.path.join(self.x_dir, self.x_list[idx])
        img = np.load(img_name)

        label_name = os.path.join(self.y_dir, self.y_list[idx])
        label = np.load(label_name)

        label = (label > 0).astype(float)

        if callable(self.tfms):
            seed = random.randint(0, 2**32)
            random.seed(seed)
            img = self.tfms(img.transpose(1, 2, 0)).transpose(2, 0, 1)

            random.seed(seed)
            label = self.tfms(label.transpose(1, 2, 0)).transpose(2, 0, 1)

        img_tensor = torch.Tensor(img)
        label_tensor = torch.Tensor(label)
        return img_tensor, label_tensor
    