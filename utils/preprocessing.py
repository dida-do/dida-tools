"""Data augmentation and preprocessing functions."""

from typing import Callable, Iterable
from torchvision import transforms
import numpy as np

from albumentations import (
    ShiftScaleRotate, CLAHE, RandomRotate90, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

def augmentation(p: float=0.5) -> Callable:
    """Data augmentation pipeline.
    
    Fairly basic, can be improved. It would also be nice to define add this to hparams so we can include them in hyperparameter searches"""
    
    return Compose([
        RandomRotate90(),
        Flip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            IAASharpen(),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=p)

def get_preprocess(n_ch: int=3,
                   base_mean: Iterable[float]=(0.485, 0.456, 0.406),
                   base_std: Iterable[float]=(0.229, 0.224, 0.225)) -> Callable:
    """Get a preprocessing pipeline given base normalisation values and the number of channels.
    
    The default mean and standard deviations are for torchvision models https://pytorch.org/docs/stable/torchvision/models.html"""
    
    base_ch = len(base_mean)
    base_mean = list(base_mean)
    base_std = list(base_std)
    if len(base_std) != base_ch:
        raise InputError("base_mean and base_std must have the same length")
    
    if n_ch == base_ch:
        mean = base_mean
        std = base_std
    elif n_ch < base_ch:
        mean = base_mean[:n_ch]
        std = base_std[:n_ch]
    elif n_ch > base_ch:
        mean = base_mean + [np.mean(base_mean)] * (n_ch - base_ch)
        std = base_std + [np.mean(base_std)] * (n_ch - base_ch)
    else:
        raise TypeError("Invalid number of channels")
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])