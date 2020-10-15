#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This is the primary fast image transformer definition trained to quickly
    transfer a style onto a photo
    Note: to make quantization work, need to add
        1) QuantStub, DeQuantStub
        2) res connections (add) -> nn.quantized.FloatFunctional 
"""

import torch
from layers import Conv, Conv1stLayer, UpConv, UpConvUS, ResLayer, ResShuffleLayer
from layers import DWSConv, DWSConvT, ConvBNReLU
from torch.quantization import fuse_modules, QuantStub, DeQuantStub


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
    def __init__(self,leak=0,
                 norm_type='batch',
                 DWS=True,DWSFL=False,
                 outerK=3,resgroups=1,
                 filters=[8,16,16],
                 shuffle=False,
                 blocks=[2,2,2,1,1],
                 endgroups=(1,1),
                 upkern=3,
                 bias_ll=True,
                 quant=False):
        super().__init__()
        self.fused = False
        self.leak = leak
        self.norm_type = norm_type
        
        # downward conv block (shrink to 1/4x1/4 image)
        self.down_conv = torch.nn.Sequential(
            Conv1stLayer(3, filters[0], outerK, 1, DWS=DWSFL, 
                         norm_type=norm_type, leak=leak),
            Conv(filters[0],filters[1], 3, 2, DWS=DWS,groups=endgroups[0],
                 norm_type=norm_type, leak=leak),
            Conv(filters[1], filters[2], 3, 2, DWS=DWS,groups=endgroups[1],
                 norm_type=norm_type, leak=leak)
        )
        
        # resblock - most effort is here
        if shuffle:
            self.res_block = torch.nn.Sequential()
            i=0
            for block in blocks:
                self.res_block.add_module(str(i),ResShuffleLayer(filters[2],
                                                          leak=leak,
                                                          norm_type=norm_type,
                                                          DWS=DWS,
                                                          groups=resgroups,
                                                          dilation=block))
                i += 1
        else:
            self.res_block = torch.nn.Sequential()
            i=0
            for block in blocks:
                self.res_block.add_module(str(i),ResLayer(filters[2],
                                                          leak=leak,
                                                          norm_type=norm_type,
                                                          DWS=DWS,
                                                          dilation=block,
                                                          groups=resgroups))
                i += 1
                
        # up conv block (grow to original size)
        if upkern == 4:
            self.up_conv = torch.nn.Sequential(
                UpConv(filters[2], filters[1], 4, 2, DWS=DWS,groups=endgroups[1],
                       norm_type=norm_type, leak=leak),
                UpConv(filters[1], filters[0], 4, 2, DWS=DWS,groups=endgroups[0],
                       norm_type=norm_type, leak=leak),
                Conv(filters[0], 3, outerK, 1, DWS=DWSFL, bias=bias_ll)
            )
        if upkern == 3:
            self.up_conv = torch.nn.Sequential(
                UpConvUS(filters[2], filters[1], 3, 2, DWS=DWS,groups=endgroups[1],
                       norm_type=norm_type, leak=leak),
                UpConvUS(filters[1], filters[0], 3, 2, DWS=DWS,groups=endgroups[0],
                       norm_type=norm_type, leak=leak),
                Conv(filters[0], 3, outerK, 1, DWS=DWSFL, bias=bias_ll)
            )  
        
        if quant:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            self.transformer = torch.nn.Sequential(self.quant,
                                                   self.down_conv,
                                                   self.res_block,
                                                   self.up_conv,
                                                   self.dequant)
        else:
            self.transformer = torch.nn.Sequential(self.down_conv,
                                                   self.res_block,
                                                   self.up_conv)
            


    def fuse(self, inplace=True):
        if self.norm_type != 'batch' or self.fused:
            print("Cannot fuse")
            return
        for m in self.modules():
            if type(m) == DWSConv or type(m) == DWSConvT:
                fuse_modules(m, ['depthwise','norm1'],inplace=inplace)
                if m.leak==0:
                    fuse_modules(m, ['pointwise','norm2','relu'],inplace=inplace) 
                else:
                    fuse_modules(m, ['pointwise','norm2'],inplace=inplace) 
            if type(m) == Conv1stLayer or type(m) == ConvBNReLU:
                if m.leak==0:
                    fuse_modules(m, ['conv2d','norm','relu'],inplace=inplace)
                else:
                    fuse_modules(m, ['conv2d','norm'],inplace=inplace)
        print("Fusion complete")
        self.fused = True

    def forward(self, ins):
        """ forward pass """
        return self.transformer(ins)