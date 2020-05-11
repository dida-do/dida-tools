#pylint: disable=R0903

"""
This module contains generic data augmentation routines for images.
It expects channels-first ordering (c, h, w) for with any number
of channels.
"""

import torch

class HorizontalFlip:
    '''
    A transformation for data augmentation that
    flips an image along horizontal axis.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img.flip([2])

class VerticalFlip:
    '''
    A transformation for data augmentation that
    flip an image along vertical axis.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img.flip([1])

class GaussianNoise:
    '''
    Add Gaussian noise to the image.
    '''
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return img + torch.empty(img.shape).normal_(mean=self.mean, std=self.std)

class Identity:
    '''
    Return the image unchanged. To be used as target transformation
    when applying Gaussian noise to input.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img

class Rotate90left:
    '''
    Rotate the image 90 degrees counter-clockwise.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img.transpose(1, 2).flip([2])

class Rotate90right:
    '''
    Rotate the image 90 degrees clockwise.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img.transpose(1, 2).flip([1])

class Rotate180:
    '''
    Rotate the image 180 degrees.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        return img.flip([1, 2])
