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
    def __init__(self,resgroups=1,expansion=6, num_levels=4, 
                 down_depth = [1,2,2,2], up_depth = [1,1,1],
                 filters=[16,24,32,48],endchannels=[16,1],groupings=(1,1),
                 upkern=3,use_JPU=False,dilate_channels=32,bias_ll=False):
        super().__init__()
        self.useJPU = False #use_JPU
        self.fused = False
        assert len(filters) == num_levels
        assert len(down_depth) == num_levels
        assert len(up_depth) == num_levels-1
        
        self.num_levels = num_levels
        
        self.encoder = [Sequential(Conv(3,filters[0],DWS=False,stride=2))]
        if upkern==3:
            self.upconv = [UpConvUS(filters[0],endchannels[0],upsample=2,DWS=True)]
        else:
            self.upconv = [UpConv(filters[0],endchannels[0],upsample=2,DWS=True)]
        self.decoder = [Sequential(InvertedResidual(2*filters[0],filters[0],
                                                    expansion))]
        
        # do level zero first
        for j in range(down_depth[0]):
            name = "DownIR_{}_{}".format(0,j)
            self.encoder[0].add_module(name,InvertedResidual(filters[0],
                                                             filters[0],
                                                             expansion))
        for j in range(up_depth[0]):
            name = "UpIR_{}_{}".format(0,j)
            self.decoder[0].add_module(name,InvertedResidual(filters[0],
                                                             filters[0],
                                                             expansion))
        # start at depth 1
        for i in range(1,num_levels):
            # build upconv layer
            if upkern==3:
                self.upconv.append(UpConvUS(filters[i],filters[i-1],
                                            upsample=2,DWS=True))
            else:
                self.upconv.append(UpConv(filters[i],endchannels[i-1],
                                          upsample=2,DWS=True))
                
            # build encoder layer
            self.encoder.append(Sequential(InvertedResidual(filters[i-1],
                                                            filters[i],
                                                            expansion,
                                                            stride=2)))
            for j in range(down_depth[i]):
                name = "DownIR_{}_{}".format(i,j)
                self.encoder[i].add_module(name,InvertedResidual(filters[i],
                                                                 filters[i],
                                                                 expansion))
            if i == num_levels - 1:
                # no decoder layer here 
                self.decoder.append(torch.nn.Identity())
            else:
                # build decoder layer
                self.decoder.append(Sequential(InvertedResidual(2*filters[i],
                                                                filters[i],
                                                                expansion)))
                for j in range(up_depth[i]):
                    name = "UpIR_{}_{}".format(i,j)
                    self.decoder[i].add_module(name,InvertedResidual(filters[i],
                                                                     filters[i],
                                                                     expansion))
        """
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
        
        if use_JPU: # note:  This is a stub
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
                self.deconv3 = UpConvUS(filters[3],filters[2],upsample=2,DWS=True)
                self.deconv2 = UpConvUS(filters[2],filters[1],upsample=2,DWS=True)
                self.deconv1 = UpConvUS(filters[1],filters[0],upsample=2,DWS=True)
                self.deconv0 = UpConvUS(filters[0],endchannels[0],upsample=2,DWS=True)
            else:
                
                self.deconv3 = UpConv(filters[3],filters[2],upsample=2,DWS=True)
                self.deconv2 = UpConv(filters[2],filters[1],upsample=2,DWS=True)
                self.deconv1 = UpConv(filters[1],filters[0],upsample=2,DWS=True)
                self.deconv0 = UpConv(filters[0],endchannels[0],upsample=2,DWS=True)
        
        self.decoder2 = Sequential(
            InvertedResidual(2*filters[2],filters[2],expansion),
            InvertedResidual(filters[2],filters[2],expansion)
            )
        self.decoder1 = Sequential(
            InvertedResidual(2*filters[1],filters[1],expansion),
            InvertedResidual(filters[1],filters[1],expansion)
            )
        self.decoder0 = Sequential(
            InvertedResidual(2*filters[0],filters[0],expansion),
            InvertedResidual(filters[0],filters[0],expansion)
            )
        """
            
        self.pred = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        self.edge = Conv(endchannels[0],endchannels[1],DWS=False,bias=bias_ll)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def cuda(self):
        super().cuda()
        for i in range(self.num_levels):
            self.encoder[i].cuda()
            self.upconv[i].cuda()
            self.decoder[i].cuda()
    
    def cpu(self):
        super().cpu()
        for i in range(self.num_levels):
            self.encoder[i].cpu()
            self.upconv[i].cpu()
            self.decoder[i].cpu()
    

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
        
        down_outs = [ self.encoder[0](xq) ]
        
        for i in range(1, num_l):
            down_outs.append(self.encoder[i](down_outs[-1]))
        
        up_outs = [None] * (num_l - 1)        
        up_outs[num_l - 2] = self.upconv[num_l - 1](down_outs[num_l - 1])
        for i in range(2, num_l):
            k = num_l - i
            temp_out = torch.cat((down_outs[k],up_outs[k]),dim=1)
            up_outs[k - 1] = self.upconv[k](self.decoder[k](temp_out))
        
        temp_out = torch.cat((down_outs[0],up_outs[0]),dim=1)
        penult = self.upconv[0](self.decoder[0](temp_out))
        
        """
        x0 = self.level0(xq)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        
        up2 = self.deconv3(x3)
        up1 = self.deconv2(self.decoder2(torch.cat((x2,up2),dim=1)))
        up0 = self.deconv1(self.decoder1(torch.cat((x1,up1),dim=1)))
        penult = self.deconv0(self.decoder0(torch.cat((x0,up0),dim=1)))
        """
    
        mask_pred = self.dequant(self.pred(penult))
        edge = self.dequant(self.edge(penult))
        
        return mask_pred, edge
