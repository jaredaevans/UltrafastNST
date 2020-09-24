#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This is the primary fast image transformer definition trained to quickly
transfer a style onto a photo
"""

import torch
from layers import Conv, Conv1stLayer, UpConv, ResLayer, ResShuffleLayer

class ImageTransformer(torch.nn.Module):
    """ This is our main model, for fast NST, currently uses:
        Deconv by factor of four
        ResLayers
        Upconv by factor of four
        LeakyReLU (0.05)
        Instance Normalization
        Note: IN may be source of water droplet features:
            https://arxiv.org/pdf/1912.04958.pdf)
    """
    def __init__(self,leak=0.05,
                 norm_type='inst',
                 DWS=True,DWSFL=False,
                 outerK=3,resgroups=4,
                 filters=[32,48,64],
                 shuffle=False,
                 endgroups=(1,1)):
        super().__init__()
        self.leak = leak
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.down_conv = torch.nn.Sequential(
            Conv1stLayer(3, filters[0], outerK, 1, DWS=DWSFL),
            norm_layer(filters[0], affine=True),
            torch.nn.LeakyReLU(leak),
            Conv(filters[0],filters[1], 3, 2, DWS=DWS,groups=endgroups[0]),
            norm_layer(filters[1], affine=True),
            torch.nn.LeakyReLU(leak),
            Conv(filters[1], filters[2], 3, 2, DWS=DWS,groups=endgroups[1]),
            norm_layer(filters[2], affine=True),
            torch.nn.LeakyReLU(leak),
        )
        if shuffle:
            self.res_block = torch.nn.Sequential(
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=1),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=2),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=1),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=2),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=1),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=2),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=1),
                ResShuffleLayer(filters[2],leak=leak,norm_type=norm_type,
                                DWS=DWS,groups=resgroups,dilation=1)
            )
        else:
            self.res_block = torch.nn.Sequential(
                ResLayer(filters[2],leak=leak,norm_type=norm_type,DWS=DWS,groups=resgroups),
                ResLayer(filters[2],leak=leak,norm_type=norm_type,DWS=DWS,groups=resgroups),
                ResLayer(filters[2],leak=leak,norm_type=norm_type,DWS=DWS,groups=resgroups),
                ResLayer(filters[2],leak=leak,norm_type=norm_type,DWS=DWS,groups=resgroups),
                ResLayer(filters[2],leak=leak,norm_type=norm_type,DWS=DWS,groups=resgroups)
            )
        self.up_conv = torch.nn.Sequential(
            UpConv(filters[2], filters[1], 4, 2, DWS=DWS,groups=endgroups[1]),
            norm_layer(filters[1], affine=True),
            torch.nn.LeakyReLU(leak),
            UpConv(filters[1], filters[0], 4, 2, DWS=DWS,groups=endgroups[0]),
            norm_layer(filters[0], affine=True),
            torch.nn.LeakyReLU(leak),
            Conv(filters[0], 3, outerK, 1, DWS=DWSFL)
        )
        self.transformer = torch.nn.Sequential(self.down_conv,
                                               self.res_block,
                                               self.up_conv)

    def forward(self, ins):
        """ forward pass """
        return self.transformer(ins)
