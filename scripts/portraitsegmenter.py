#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:17:05 2020

@author: JAE
"""

import torch
from torch.nn import Sequential

from layers import Conv, UpConv, UpConvUS, InvertedResidual
from layers import DWSConv, DWSConvT, ConvBNReLU, Layer131
from torch.quantization import fuse_modules, QuantStub, DeQuantStub


class PortraitSegmenter(torch.nn.Module):
    """ This is out UNet style segmenter.  We work on a small image, and 
        scale up the result.
    """
    def __init__(self,resgroups=1,expansion=6, num_levels=4, 
                 down_depth = [1,2,2,2], up_depth = [1,1,1],
                 filters=[16,24,32,48],endchannels=[16,1],groupings=(1,1),
                 upkern=3,use_JPU=False,dilate_channels=32,bias_ll=False):
        super().__init__()
        self.useJPU = False #use_JPU
        self.fused = False
        
        assert num_levels >= 3 and num_levels <= 6
        assert len(filters) == num_levels
        assert len(down_depth) == num_levels
        assert len(up_depth) == num_levels-1
        
        
        self.num_levels = num_levels
        
        # drop to 1/2
        self.encoder0 = Sequential(Conv(3,filters[0],DWS=False,stride=2))
        for j in range(down_depth[0]):
            name = "DownIR_{}_{}".format(0,j)
            self.encoder0.add_module(name,InvertedResidual(filters[0],
                                                           filters[0],
                                                           expansion))
        self.decoder0 = Sequential(InvertedResidual(2*filters[0],filters[0],
                                                    expansion))
        for j in range(up_depth[0]):
            name = "UpIR_{}_{}".format(0,j)
            self.decoder0.add_module(name,InvertedResidual(filters[0],
                                                           filters[0],
                                                           expansion))
        if upkern==3:
            self.upconv0 = UpConvUS(filters[0],endchannels[0],upsample=2,DWS=True)
        else:
            self.upconv0 = UpConv(filters[0],endchannels[0],upsample=2,DWS=True)
            
        # drop to 1/4
        i = 1
        self.encoder1 = Sequential(InvertedResidual(filters[i-1],
                                                    filters[i],
                                                    expansion,
                                                    stride=2))
        for j in range(down_depth[i]):
            name = "DownIR_{}_{}".format(i,j)
            self.encoder1.add_module(name,InvertedResidual(filters[i],
                                                           filters[i],
                                                           expansion))
        self.decoder1 = Sequential(InvertedResidual(2*filters[i],
                                                    filters[i],
                                                    expansion))
        for j in range(up_depth[i]):
            name = "UpIR_{}_{}".format(i,j)
            self.decoder1.add_module(name,InvertedResidual(filters[i],
                                                           filters[i],
                                                           expansion))
        if upkern==3:
            self.upconv1 = UpConvUS(filters[i],filters[i-1],upsample=2,DWS=True)
        else:
            self.upconv1 = UpConv(filters[i],filters[i-1],upsample=2,DWS=True)
        
        # drop to 1/8
        i = 2
        self.encoder2 = Sequential(InvertedResidual(filters[i-1],
                                                    filters[i],
                                                    expansion,
                                                    stride=2))
        for j in range(down_depth[i]):
            name = "DownIR_{}_{}".format(i,j)
            self.encoder2.add_module(name,InvertedResidual(filters[i],
                                                           filters[i],
                                                           expansion))
        if upkern==3:
            self.upconv2 = UpConvUS(filters[i],filters[i-1],upsample=2,DWS=True)
        else:
            self.upconv2 = UpConv(filters[i],filters[i-1],upsample=2,DWS=True)
        
        if num_levels > 3:
            # note: decoders only need one fewer level
            self.decoder2 = Sequential(InvertedResidual(2*filters[i],
                                                    filters[i],
                                                    expansion))
            for j in range(up_depth[i]):
                name = "UpIR_{}_{}".format(i,j)
                self.decoder2.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
            
            # drop to 1/16
            i = 3
            self.encoder3 = Sequential(InvertedResidual(filters[i-1],
                                                        filters[i],
                                                        expansion,
                                                        stride=2))
            for j in range(down_depth[i]):
                name = "DownIR_{}_{}".format(i,j)
                self.encoder3.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
            if upkern==3:
                self.upconv3 = UpConvUS(filters[i],filters[i-1],upsample=2,DWS=True)
            else:
                self.upconv3 = UpConv(filters[i],filters[i-1],upsample=2,DWS=True)
                
        if num_levels > 4:
            self.decoder3 = Sequential(InvertedResidual(2*filters[i],
                                                        filters[i],
                                                        expansion))
            for j in range(up_depth[i]):
                name = "UpIR_{}_{}".format(i,j)
                self.decoder3.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
            
            # drop to 1/32
            i = 4
            self.encoder4 = Sequential(InvertedResidual(filters[i-1],
                                                        filters[i],
                                                        expansion,
                                                        stride=2))
            for j in range(down_depth[i]):
                name = "DownIR_{}_{}".format(i,j)
                self.encoder4.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
            if upkern==3:
                self.upconv4 = UpConvUS(filters[i],filters[i-1],upsample=2,DWS=True)
            else:
                self.upconv4 = UpConv(filters[i],filters[i-1],upsample=2,DWS=True)
        
        if num_levels > 5:
            self.decoder4 = Sequential(InvertedResidual(2*filters[i],
                                                        filters[i],
                                                        expansion))
            for j in range(up_depth[i]):
                name = "UpIR_{}_{}".format(i,j)
                self.decoder4.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
                
            # drop to 1/64
            i = 5
            self.encoder5 = Sequential(InvertedResidual(filters[i-1],
                                                        filters[i],
                                                        expansion,
                                                        stride=2))
            for j in range(down_depth[i]):
                name = "DownIR_{}_{}".format(i,j)
                self.encoder5.add_module(name,InvertedResidual(filters[i],
                                                               filters[i],
                                                               expansion))
            if upkern==3:
                self.upconv5 = UpConvUS(filters[i],filters[i-1],upsample=2,DWS=True)
            else:
                self.upconv5 = UpConv(filters[i],filters[i-1],upsample=2,DWS=True)
        
            
        self.pred = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        self.edge = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()    

    def fuse(self, inplace=True):
        if self.fused:
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
        num_l = self.num_levels
        
        xq = self.quant(ins)

        x0 = self.encoder0(xq)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        if num_l > 3:
            x3 = self.encoder3(x2)
            if num_l > 4:
                x4 = self.encoder4(x3)
                if num_l > 5:
                    x5 = self.encoder5(x4)
                    
                    up4 = self.upconv5(x5)
                    inp4 = self.decoder4(torch.cat((x4,up4),dim=1))
                else:
                    inp4 = x4
                up3 = self.upconv4(inp4)
                inp3 = self.decoder3(torch.cat((x3,up3),dim=1))
            else:
                inp3 = x3
            up2 = self.upconv3(inp3)
            inp2 = self.decoder2(torch.cat((x2,up2),dim=1))
        else:
            inp2 = x2
        up1 = self.upconv2(inp2)
        inp1 = self.decoder1(torch.cat((x1,up1),dim=1))
        
        up0 = self.upconv1(inp1)
        penult = self.upconv0(self.decoder0(torch.cat((x0,up0),dim=1)))
    
        mask_pred = self.dequant(self.pred(penult))
        edge = self.dequant(self.edge(penult))
        
        return mask_pred, edge
