"""
U-NET Architecture. We recursively build modules that downsample and upsample
before concatenating with the copied input. Options exist for specifying the
exact architecture required.
"""

from typing import Optional

from torch import nn
import torch

from layers.conv import ConvLayer, DeconvLayer, TransConvLayer

class UnetBulk(nn.Module):
    """
    Recursively build a U-NET
    """
    def __init__(self, ch_in: int = 32, n_recursions: int = 4, dropout: Optional[float] = None,
                 use_shuffle: bool = True, activ: nn.Module = nn.ELU, use_pooling: bool = False):
        """
        :param ch_in: number of input Channels
        :param n_recursions: number of times to repeat
        :param use_shuffle: whether to use pixel shuffel or traditional deconvolution
        :param dropout: whether or not to use dropout and how much
        :param use_pooling: whether to use pooling or strides for downsampling
        """
        super().__init__()
        layers = [ConvLayer(ch_in, ch_in, stride=2-int(use_pooling), dropout=dropout, activ=activ),
                  ConvLayer(ch_in, ch_in, dropout=dropout, activ=activ),
                  ConvLayer(ch_in, 2 * ch_in, dropout=dropout, activ=activ)]


        if use_pooling:
            layers.insert(1, nn.MaxPool2d(2))

        self.down = nn.Sequential(*layers)

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
    """
    Wrapper for UNET_Bulk together with input and output layers
    """
    def __init__(self, ch_in: int = 12, ch_out: int = 2, bulk_ch: int = 32, n_recursions: int = 4,
                 use_shuffle: bool = True, dropout: Optional[float] = None,
                 activ: nn.Module = nn.ELU, use_pooling: bool = True):
        """
        :param ch_in: number of input Channels
        :param ch_out: number of output Channels
        :param bulk_ch: initial channels for bulk
        :param n_recursions: number of times to repeat
        :param use_shuffle: whether to use pixel shuffel or traditional deconvolution
        :param dropout: whether or not to use dropout and how much
        """
        super().__init__()
        self.in_layer = ConvLayer(ch_in, bulk_ch, ks=1, pad=0, dropout=dropout, activ=activ)

        self.bulk = UnetBulk(ch_in=bulk_ch, n_recursions=n_recursions,
                             dropout=dropout, use_shuffle=use_shuffle,
                             activ=activ, use_pooling=use_pooling)

        self.out = nn.Conv2d(2 * bulk_ch, ch_out, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        x = self.bulk(x)
        return self.out(x)
