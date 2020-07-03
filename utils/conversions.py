from typing import Callable
from dataclasses import make_dataclass, field
import inspect
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

def snake2camal(string: str) -> str:
    """
    Change string from snake_case to CamalCase

    :param string: (str) string to convert

    :rtype (str): converted string
    """
    return ''.join(word.title() for word in string.split('_'))

def objectise(fn: Callable) -> type:
    """
    Take a function and create a class with kwargs as attributes and a
    call method that evaluates the function.

    :param fn: (Callable) function to convert to class

    :rtype (dataclass): callable class with kwargs of fn as attributes
    """
    field_list = []
    for k, v in inspect.signature(fn).parameters.items():
        if v.default is not inspect.Parameter.empty:
            field_list.append((k, v.annotation, field(default=v.default)))

    def wrap_call(self, *args):
        return fn(*args, **{k: self.__dict__[k] for k, _, _ in field_list})

    return make_dataclass(snake2camal(fn.__name__), field_list, namespace={"__call__": wrap_call})
