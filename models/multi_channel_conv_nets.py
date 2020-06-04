"""Models for multi channel image data.

Mostly this is a wrapper for torchvision models but it also supports our unet implementation from the deep learning repo."""

import sys
sys.path.append('../')

import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from models.unet import UNET

# This could be implemented using the module.__dict__ attr, not sure if that is nicer...
SEG_MODELS = {
    "deeplabv3": models.segmentation.deeplabv3_resnet101,
    "fcn": models.segmentation.fcn_resnet101,
    "unet": UNET
}

CLS_MODELS = {
    "resnet18": models.resnet18,
    "resnet101": models.resnet101
}

def modify_conv(conv_layer: nn.Module, ch_in: int) -> nn.Module:
    """Take a convolutional layer and modify the input channels, keeping weights as much as possible
    
    Largely taken from github.com/sshuair/torchsat"""
    
    if ch_in == conv_layer.in_channels:
        return conv_layer
    
    new_conv = nn.Conv2d(in_channels=ch_in,
                         out_channels=conv_layer.out_channels,
                         kernel_size=conv_layer.kernel_size,
                         stride=conv_layer.kernel_size,
                         padding=conv_layer.padding,
                         bias=(conv_layer.bias != None))
    
    new_conv.bias = conv_layer.bias
    
    if ch_in < conv_layer.in_channels:
        new_conv.weight.data = conv_layer.weight.data[:, :ch_in, :, :]
    elif ch_in > conv_layer.in_channels:
        multi = ch_in // conv_layer.in_channels
        last = ch_in % conv_layer.in_channels
        new_conv.weight.data[:, :conv_layer.in_channels * multi, :, :] = torch.cat([conv_layer.weight.data for x in range(multi)], dim=1)
        new_conv.weight.data[:, conv_layer.in_channels * multi:, :, :] = conv_layer.weight.data[:, :last, :, :]
    
    return new_conv

class P2PModel(nn.Module):
    def __init__(self, model_name: str="deeplabv3", ch_in: int=3, ch_out: int=3, **kwargs):
        """Return a segmentation model (pixel to pixel)
        
        Would need modification for non torchvision Resnent based model"""
        
        super().__init__()
        
        if model_name not in SEG_MODELS:
            raise NotImplementedError(f"Model {model_name} not currently implemented")
        
        # The unet is a special case
        if model_name == "unet":
            if "n_recursions" not in kwargs:
                kwargs["n_recursions"] = 5
            if "dropout" not in kwargs:
                kwargs["dropout"] = 0.1
            
            self.model = UNET(ch_in, ch_out, **kwargs)
        else:
            model = SEG_MODELS[model_name](pretrained=True)
        
            model.backbone.conv1 = modify_conv(model.backbone.conv1, ch_in)
            model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, ch_out, kernel_size=(1, 1), stride=(1, 1))
        
            self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        
        if type(output) in [dict, OrderedDict]:
            return output["out"]
        else:
            return output
    
def binary_clf_model(model_name: str="resnet18", ch_in: int=3) -> nn.Module:
    """Return a binary classification model
    
    Would need modification for non torchvision resnet model"""
    if model_name not in CLS_MODELS:
        raise NotImplementedError(f"Model {model_name} not currently implemented")
    
    model = CLS_MODELS[model_name](pretrained=True)
    model.conv1 = modify_conv(model.conv1, ch_in)
    model.fc = nn.Linear(model.fc.in_features, 1, bias=True)
    
    return model