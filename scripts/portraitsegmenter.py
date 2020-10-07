#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:17:05 2020

@author: JAE
"""

import torch
from torch.nn import Sequential

from layers import Conv, UpConv, UpConvUS, InvertedResidual, ResLayer
from layers import DWSConv, DWSConvT, ConvBNReLU, Layer131
from torch.quantization import fuse_modules, QuantStub, DeQuantStub


class PortraitSegmenter(torch.nn.Module):
    """ This is out UNet style segmenter.  We work on a small image, and 
        scale up the result.
    """
    def __init__(self,resgroups=1,expansion=6, dilate_channels=32,
                 filters=[16,24,32,48],endchannels=[16,1],groupings=(1,1),
                 upkern=3,use_JPU=False,bias_ll=False):
        super().__init__()
        self.useJPU = use_JPU
        # drop to 1/2
        self.level0 = Sequential(
            Conv(3,filters[0],DWS=False,stride=2),
            InvertedResidual(filters[0],filters[0],expansion)
            )
        # drop to 1/4
        self.level1 = Sequential(
            InvertedResidual(filters[0],filters[1],expansion,stride=2),
            InvertedResidual(filters[1],filters[1],expansion),
            InvertedResidual(filters[1],filters[1],expansion)
            )
        
        # drop to 1/8
        self.level2 = Sequential(
            InvertedResidual(filters[1],filters[2],expansion,stride=2),
            InvertedResidual(filters[2],filters[2],expansion),
            InvertedResidual(filters[2],filters[2],expansion)
            )
        
        # drop to 1/16
        self.level3 = Sequential(
            InvertedResidual(filters[2],filters[3],expansion,stride=2),
            InvertedResidual(filters[3],filters[3],expansion),
            InvertedResidual(filters[3],filters[3],expansion)
            )
        
        if use_JPU:
            self.dilation1 = DWSConv(dilate_channels,dilate_channels,3, 
                                     dilation=1,bias=False,pad=True)
            self.dilation1 = DWSConv(dilate_channels,dilate_channels,3, 
                                     dilation=2,bias=False,pad=True)
            self.dilation1 = DWSConv(dilate_channels,dilate_channels,3, 
                                     dilation=4,bias=False,pad=True)
            self.dilation1 = DWSConv(dilate_channels,dilate_channels,3, 
                                     dilation=8,bias=False,pad=True)
        else:
            if upkern==3:
                self.res2 = ResLayer(filters[2],filters[2],DWS=True,leak=0)
                self.res1 = ResLayer(filters[1],filters[1],DWS=True,leak=0)
                self.res0 = ResLayer(filters[0],filters[0],DWS=True,leak=0)
                
                self.deconv3 = UpConvUS(filters[3],filters[2],upsample=2,DWS=True)
                self.deconv2 = UpConvUS(2*filters[2],filters[1],upsample=2,DWS=True)
                self.deconv1 = UpConvUS(2*filters[1],filters[0],upsample=2,DWS=True)
                self.deconv0 = UpConvUS(2*filters[0],endchannels[0],upsample=2,DWS=True)
            else:
                self.res2 = ResLayer(filters[2],filters[2],DWS=True,leak=0)
                self.res1 = ResLayer(filters[1],filters[1],DWS=True,leak=0)
                self.res0 = ResLayer(filters[0],filters[0],DWS=True,leak=0)
                
                self.deconv3 = UpConv(filters[3],filters[2],upsample=2,DWS=True)
                self.deconv2 = UpConv(2*filters[2],filters[1],upsample=2,DWS=True)
                self.deconv1 = UpConv(2*filters[1],filters[0],upsample=2,DWS=True)
                self.deconv0 = UpConv(2*filters[0],endchannels[0],upsample=2,DWS=True)
            
        self.pred = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        self.edge = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    

    def fuse(self, inplace=True):
        if self.norm_type != 'batch' or self.fused:
            print("Cannot fuse")
            return
        for m in self.modules():
            if type(m) == Layer131:
                fuse_modules(m, ['pointwise','norm3'],inplace=inplace) 
                if m.leak==0:
                    fuse_modules(m, ['firstlayer','norm1','relu1'],inplace=inplace) 
                    fuse_modules(m, ['depthwise','norm2','relu2'],inplace=inplace)
                else:
                    fuse_modules(m, ['firstlayer','norm1'],inplace=inplace) 
                    fuse_modules(m, ['depthwise','norm2'],inplace=inplace)
            if type(m) == DWSConv or type(m) == DWSConvT:
                fuse_modules(m, ['depthwise','norm1'],inplace=inplace)
                if m.leak==0:
                    fuse_modules(m, ['pointwise','norm2','relu'],inplace=inplace) 
                else:
                    fuse_modules(m, ['pointwise','norm2'],inplace=inplace) 
            if type(m) == ConvBNReLU:
                if m.leak==0:
                    fuse_modules(m, ['conv2d','norm','relu'],inplace=inplace)
                else:
                    fuse_modules(m, ['conv2d','norm'],inplace=inplace)
        print("Fusion complete")
        self.fused = True


    def forward(self, ins):
        """ forward pass """
        xq = self.quant(ins)
        x0 = self.level0(xq)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        
        up2 = self.deconv3(x3)
        up1 = self.deconv2(torch.cat((self.res2(x2),up2),dim=1))
        up0 = self.deconv1(torch.cat((self.res1(x1),up1),dim=1))
        
        penult = self.deconv0(torch.cat((self.res0(x1),up0),dim=1))
        mask_pred = self.dequant(self.pred(penult))
        edge = self.dequant(self.edge(penult))
        
        return mask_pred, edge
