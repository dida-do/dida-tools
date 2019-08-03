"""
Contains custom convolutional layers
"""

from typing import Optional
from torch import nn
import torch

class ConvLayer(nn.Module):
    '''
    Standard Convolutional Layer.
    Combining convolution, activation and batchnorm
    '''
    def __init__(self, ch_in: int, ch_out: int, pad: int=1,
                 ks: int=3, stride: int=1, dropout: Optional[float]=None,
                 activ: nn.Module=nn.ELU):
        '''
        :param ch_in: (int) number of input Channels
        :param: ch_out: (int) number of output Channels

        :param pad: (int) number of pixels to pad by
        :param ks: (int) kernel size
        :param stride: (int) stride for convolution
        :param dropout: (float or none) whether or not to use
            dropout and how much
        '''
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.conv = nn.Conv2d(ch_in, ch_out, (ks, ks), padding=pad, stride=stride, bias=False)
        self.activ = activ(inplace=True)
        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if callable(self.dropout):
            x = self.dropout(x)

        x = self.conv(x)
        x = self.activ(x)
        x = self.norm(x)
        return x

class DeconvLayer(nn.Module):
    '''
    Standard deconvolution layer, uses pixel shuffle
    https://arxiv.org/abs/1707.02937
    '''
    def __init__(self, ch_in: int, ch_out: int,
                 stride: int=2, dropout: Optional[float]=None,
                 activ: nn.Module=nn.ELU):
        '''
        :param ch_in: (int) number of input Channels
        :param ch_out: (int) number of output Channels

        :param stride: (int) stride for deconvolution
        :param dropout: (float or none) whether or not to use dropout
            and how much
        '''
        super().__init__()
        self.conv = ConvLayer(ch_in, (stride**2) * ch_out, dropout=dropout, activ=activ)
        self.pixelshuffle = nn.PixelShuffle(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.pixelshuffle(x)

class TransConvLayer(nn.Module):
    '''
    Standard Convolutional Layer.
    Combining convolution, activation and batchnorm
    '''
    def __init__(self, ch_in: int, ch_out: int,
                 pad: int=1, ks: int=3, stride: int=2,
                 dropout: Optional[float]=None,
                 activ: nn.Module=nn.ELU):
        '''
        :param ch_in: (int) number of input Channels
        :param ch_out: (int) number of output Channels

        :param pad: (int) number of pixels to pad by
        :param ks: (int) kernel size
        :param stride: (int) stride for convolution
        :param dropout: (float or none) whether or not to use
            dropout and how much
        '''
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.conv = nn.ConvTranspose2d(ch_in, ch_out, (ks, ks), padding=pad, stride=stride, bias=False)
        self.activ = activ(inplace=True)
        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if callable(self.dropout):
            x = self.dropout(x)

        x = self.conv(x)
        x = self.activ(x)
        x = self.norm(x)
        return x