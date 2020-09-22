#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Layers used in the image transformer
"""

import torch

class ReflectPad2d(torch.nn.Module):
    """ reflectionpad2d that can be transfered across onnx etc
        size : int (the size of padding)
    """
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, ins):
        size = self.size
        l_list, r_list = [], []
        u_list, d_list = [], []
        for i in range(size):#i:0, 1
            left = ins[:, :, :, (size-i):(size-i+1)]
            l_list.append(left)
            right = ins[:, :, :, (i-size-1):(i-size)]
            r_list.append(right)
        l_list.append(ins)
        ins = torch.cat(l_list+r_list[::-1], dim=3)
        for i in range(size):
            up = ins[:, :, (size-i):(size-i+1), :]
            u_list.append(up)
            down = ins[:, :, (i-size-1):(i-size), :]
            d_list.append(down)
        u_list.append(ins)
        ins = torch.cat(u_list+d_list[::-1], dim=2)
        return ins

class DWSConv(torch.nn.Module):
    """ Depthwise separable convolution: splits into a depthwise
        and pointwise conv with no activation function in between
        optional grouping on the pointwise conv
    """
    def __init__(self, ins, outs, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(ins, ins, kernel_size, stride=stride, groups=ins)
        self.pointwise = torch.nn.Conv2d(ins, outs, 1, groups=groups)

    def forward(self, ins):
        """ forward pass """
        return self.pointwise(self.depthwise(ins))


class DWSConvT(torch.nn.Module):
    """ Depthwise separable convolution transpose: splits into a depthwise
        and pointwise conv with no activation function in between
        optional grouping on the pointwise conv, grows to a larger size via
        stride
        Note: upsampling would be better, but issues with onnx / mlmodel
        padding and conversion
    """
    def __init__(self, ins, outs, kernel_size=4, stride=2, padding=1, groups=1):
        super().__init__()
        self.depthwise = torch.nn.ConvTranspose2d(ins, ins, kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=ins)
        self.pointwise = torch.nn.Conv2d(ins, outs, 1, groups=groups)

    def forward(self, ins):
        """ forward pass """
        return self.pointwise(self.depthwise(ins))


class Swish(torch.nn.Module):
    """ Swish activation function """
    def __init__(self):
        super().__init__()

    def forward(self, ins):
        """ forward pass """
        return ins * torch.sigmoid(ins)


class Conv(torch.nn.Module):
    """ Conv layer with reflection or zero padding """
    def __init__(self,ins,outs,kernel_size=3,stride=1,padding='ref',
                 DWS=False,groups=1):
        assert kernel_size % 2 == 1
        super().__init__()
        padding_size = kernel_size // 2
        #self.pad = torch.nn.ReflectionPad2d(padding_size)
        self.pad = ReflectPad2d(padding_size)
        if padding == 'zero':
            self.pad = torch.nn.ZeroPad2d()(padding_size)
        self.conv2d = torch.nn.Conv2d(ins, outs, kernel_size, stride)
        if DWS:
            self.conv2d = DWSConv(ins, outs, kernel_size, stride, groups=groups)

    def forward(self, ins):
        """ forward pass """
        out = self.pad(ins)
        out = self.conv2d(out)
        return out


class UpConv(torch.nn.Module):
    """ Up sample conv layer with zero padding
        note: conv transpose is used as this is compatible with coremltools
        worse for checkerboard artifacts - revisit with coreml version update
    """
    def __init__(self, ins, outs, kernel_size=4, upsample=2, padding='ref',
                 DWS=False, groups=1):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(ins, outs, kernel_size, stride=upsample, padding=1)
        if DWS:
            self.conv2d = DWSConvT(ins, outs, kernel_size, stride=upsample,
                                   padding=1, groups=groups)

    def forward(self, ins):
        """ forward pass """
        return self.conv2d(ins)


class SE(torch.nn.Module):
    """ Squeeze and excite layer
    """
    def __init__(self, channels):
        super().__init__()
        self.conv2d = torch.nn.AvgPool2d()

    def forward(self, ins):
        """ forward pass """
        return self.conv2d(ins)


class ResLayer(torch.nn.Module):
    """ Basic residual layer to import into ImageTransformer
    """
    def __init__(self,channels,kernel_size=3,leak=0.05,norm_type='batch',
                 DWS=False, groups=1):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.conv1 = Conv(channels,channels,kernel_size,DWS=DWS, groups=groups)
        self.norm1 = norm_layer(channels, affine=True)
        self.relu = torch.nn.LeakyReLU(leak)
        self.conv2 = Conv(channels,channels,kernel_size,DWS=DWS)
        self.norm2 = norm_layer(channels, affine=True)

    def forward(self, ins):
        """ forward pass """
        res = ins
        out = self.relu(self.norm1(self.conv1(ins)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + res)


class ResLayer1(torch.nn.Module):
    """ 1-3-1 residual layer to import into ImageTransformer
    """
    def __init__(self,channels,kernel_size=3,leak=0.05,norm_type='batch',
                 DWS=False,groups=1):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.conv1 = Conv(channels,channels,kernel_size,DWS=DWS, groups=groups)
        self.norm1 = norm_layer(channels, affine=True)
        self.relu = torch.nn.LeakyReLU(leak)
        self.conv2 = Conv(channels,channels,kernel_size,DWS=DWS)
        self.norm2 = norm_layer(channels, affine=True)

    def forward(self, ins):
        """ forward pass """
        res = ins
        out = self.relu(self.norm1(self.conv1(ins)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + res)


class ResBottle(torch.nn.Module):
    """ MobileNetv2 style residual linear bottleneck layer
        to import into ImageTransformer
    """
    def __init__(self,channels,leak=0.05,norm_type='batch',
                 DWS=True,groups=1):
        super().__init__()
        self.layer1 = ResLayer1(channels,leak=leak,norm_type=norm_type,DWS=DWS,groups=groups)

    def forward(self, ins):
        """ forward pass """
        res = ins
        # upsample, then apply conv
        out = self.layer1(ins)
        return self.relu(out + res)
