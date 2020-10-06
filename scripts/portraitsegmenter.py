#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:17:05 2020

@author: JAE
"""

import torch
from torch.nn import Sequential, ReLU
from torch.nn import BatchNorm2d as BN

from layers import Conv, UpConv, UpConvUS, InvertedResidual


class PortraitSegmenter(torch.nn.Module):
    """ This is out UNet style segmenter.  We work on a small image, and 
        scale up the result.
    """
    def __init__(self,resgroups=1,expansion=6,
                 filters=[16,24,32,48],endchannels=[8,1],groupings=(1,1),
                 upkern=3,bias_ll=False):
        super().__init__()
        # drop to 1/2
        self.level0 = Sequential(
            Conv(3,filters[0],DWS=False,stride=2),
            BN(filters[0],affine=True),
            ReLU(inplace=True),
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
        
        if upkern==3:
            self.deconv3 = UpConvUS(filters[3],filters[2],upsample=2,DWS=True)
            self.deconv2 = UpConvUS(filters[2],filters[1],upsample=2,DWS=True)
            self.deconv1 = UpConvUS(filters[1],filters[0],upsample=2,DWS=True)
            self.deconv0 = UpConvUS(filters[0],endchannels[0],upsample=2,DWS=True)
        else:
            self.deconv3 = UpConv(filters[3],filters[2],upsample=2,DWS=True)
            self.deconv2 = UpConv(filters[2],filters[1],upsample=2,DWS=True)
            self.deconv1 = UpConv(filters[1],filters[0],upsample=2,DWS=True)
            self.deconv0 = UpConv(filters[0],endchannels[0],upsample=2,DWS=True)
            
        self.pred = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        self.edge = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
    
    """
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
            if type(m) == Conv1stLayer:
                if m.leak==0:
                    fuse_modules(m, ['conv2d','norm','relu'],inplace=inplace)
                else:
                    fuse_modules(m, ['conv2d','norm'],inplace=inplace)
        print("Fusion complete")
        self.fused = True
        """
    
    def forward(self, ins):
        """ forward pass """
        x0 = self.level0(ins)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        
        up2 = self.deconv3(x3)
        up1 = self.deconv2(x2+up2)
        up0 = self.deconv1(x1+up1)
        
        penult = self.deconv0(x0+up0)
        
        return self.pred(penult), self.edge(penult)
        
        
        return self.transformer(ins)
