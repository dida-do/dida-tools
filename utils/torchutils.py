"""
Contains generic pytorch model operations such as loading,
forward and backward passes and tensor evaluations.
"""

import numpy as np
from torch import nn
import torch

#TODO add typing and sphinx docstrings

def set_seed(seed: int):
    """
    Sets random seed for numpy and both CPU/GPU pytorch routines.
    If cuda is not available, this is silently ignored.
    Note: numpy and pytorch do NOT use the same RNG.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model(model: nn.Module, model_args: dict, model_weights: str, device: torch.device):
    """
    Given model class, instantiation arguments and weight file, load the model
    """
    model = model(**model_args)
    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict["model"])
    return model

def forward(model: nn.Module, inputs: torch.Tensor, device: torch.device):
    """
    Performs a feed forward step of the
    given model and returns model outputs given a torch.Tensor input.
    """

    model.eval()
    model.to(device)

    with torch.no_grad():
        inputs = inputs.to(device)
        return model(inputs)

def backprop(model, loss_function, optimizer, batch, device):
    """
    Performs a single step of backpropagation
    and weight optimization
    """
    model.train()
    model.to(device)
    optimizer.zero_grad()

    inputs, targets = batch[0], batch[1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()

def assert_not_nan(tensor):
    assert not torch.isnan(tensor).any()

def assert_isfinite(tensor):
    assert torch.isfinite(tensor).all()

def assert_upperbound(tensor, val):
    assert (tensor < val).all()

def assert_lowerbound(tensor, val):
    assert (tensor > val).all()

def assert_nonzero(tensor):
    assert (tensor != 0).all()
    
def set_module_trainable(module: nn.Module, mode: bool) -> None:
    """Freeze or unfreeze weights"""
    for param in module.parameters():
        param.requires_grad = mode
    