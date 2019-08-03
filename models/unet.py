"""
**models/unet.py**

U-NET Architecture. We recursively build modules that downsample and upsample
before concatinating with the copied input. Options exist for specifying the
exact architecture reqiuired.

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

class UnetBulk(nn.Module):
    '''
    Recursively build a U-NET
    '''
    def __init__(self, ch_in: int=32, n_recursions: int=4, dropout: Optional[float]=None,
                 use_shuffle :bool=True, activ: nn.Module=nn.ELU):
        super().__init__()
        '''
        :param ch_in: (int) number of input Channels
        :param n_recursions: (int) number of times to repeat
        :param use_shuffle: (bool) whether to use pixel shuffel or traditional deconvolution
        :param dropout: (float or none) whether or not to use dropout and how much
        '''
        self.down = nn.Sequential(ConvLayer(ch_in, ch_in, stride=2, dropout=dropout, activ=activ),
                                  ConvLayer(ch_in, ch_in, dropout=dropout, activ=activ),
                                  ConvLayer(ch_in, 2 * ch_in, dropout=dropout, activ=activ))

        if n_recursions > 1:
            self.rec_unet = UnetBulk(ch_in=2*ch_in,
                                     n_recursions=n_recursions-1,
                                     dropout=dropout,
                                     use_shuffle=use_shuffle,
                                     activ=activ)
            down_chs = 4 * ch_in
        else:
            self.rec_unet = lambda x: x
            down_chs = 2 * ch_in

        if use_shuffle:
            deconv = DeconvLayer(ch_in, ch_in, dropout=dropout, activ=activ)
        else:
            deconv = TransConvLayer(ch_in, ch_in, dropout=dropout, activ=activ)

        self.up = nn.Sequential(ConvLayer(down_chs, ch_in, dropout=dropout, activ=activ),
                                ConvLayer(ch_in, ch_in, dropout=dropout, activ=activ),
                                deconv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_path = self.down(x)
        down_path = self.rec_unet(down_path)
        return torch.cat([self.up(down_path), x], dim=1)

class UNET(nn.Module):
    '''
    Wrapper for UNET_Bulk together with input and output layers
    '''
    def __init__(self, ch_in: int=12, ch_out: int=2, bulk_ch: int=32, n_recursions: int=4,
                 use_shuffle: bool=True, dropout: Optional[float]=None, activ: nn.Module=nn.ELU):
        super().__init__()
        '''
        :param ch_in: (int) number of input Channels
        :param ch_out: (int) number of output Channels
        :param bulk_ch: (int) initial channels for bulk
        :param n_recursions: (int) number of times to repeat
        :param use_shuffle: (bool) whether to use pixel shuffel or traditional deconvolution
        :param dropout: (float or none) whether or not to use dropout and how much
        '''
        self.in_layer = ConvLayer(ch_in, bulk_ch, ks=1, pad=0, dropout=dropout, activ=activ)

        self.bulk = UnetBulk(ch_in=bulk_ch, n_recursions=n_recursions,
                             dropout=dropout, use_shuffle=use_shuffle,
                             activ=activ)

        self.out = nn.Conv2d(2 * bulk_ch, ch_out, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        x = self.bulk(x)
        return self.out(x)
