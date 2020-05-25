import numpy as np
import PIL
import torch

def to_ndarray(image):
    """
    Convert torch.Tensor or PIL.Image.Image to ndarray.

    :param image: (torch.Tensor or PIL.Image.Image) image to convert to ndarray

    :rtype (ndarray): image as ndarray
    """
    if isinstance(image, torch.Tensor):
        return image.numpy()
    if isinstance(image, PIL.Image.Image):
        return np.array(image)
    raise TypeError("to_ndarray: expect torch.Tensor or PIL.Image.Image")
