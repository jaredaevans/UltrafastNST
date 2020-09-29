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


def shuffle_v1(inputs,group):
    """ Shuffle inputs for shuffle net v1 block """
    batchsize, num_channels, height, width = inputs.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group    
    inputs = inputs.reshape(batchsize, group_channels, group, height, width)
    inputs = inputs.permute(0, 2, 1, 3, 4)
    inputs = inputs.reshape(batchsize, num_channels, height, width)
    return inputs
    
    
def shuffle_v2(inputs):
    """ Shuffle inputs for shuffle net v2 block """
    batchsize, num_channels, height, width = inputs.data.size()
    assert (num_channels % 4 == 0)
    inputs = inputs.reshape(batchsize * num_channels // 2, 2, height * width)
    inputs = inputs.permute(1, 0, 2)
    inputs = inputs.reshape(2, -1, num_channels // 2, height, width)
    return inputs[0], inputs[1]


class DWSConv(torch.nn.Module):
    """ Depthwise separable convolution: splits into a depthwise
        and pointwise conv with no activation function in between
        optional grouping on the pointwise conv
    """
    def __init__(self,ins,outs,kernel_size=3,stride=1,groups=1,dilation=1,
                 norm_type='batch',bias=False):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.norm = norm_layer(ins, affine=True)
        self.depthwise = torch.nn.Conv2d(ins, ins, kernel_size, stride=stride, 
                                         groups=ins, dilation=dilation, 
                                         bias=False)
        self.pointwise = torch.nn.Conv2d(ins, outs, 1, groups=groups, 
                                         bias=bias)

    def forward(self, ins):
        """ forward pass """
        return self.pointwise(self.norm(self.depthwise(ins)))


class DWSConvT(torch.nn.Module):
    """ Depthwise separable convolution transpose: splits into a depthwise
        and pointwise conv with no activation function in between
        optional grouping on the pointwise conv, grows to a larger size via
        stride
        Note: upsampling would be better, but issues with onnx / mlmodel
        padding and conversion
    """
    def __init__(self,ins,outs,kernel_size=4,stride=2,padding=1,
                 groups=1,norm_type='batch'):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.norm = norm_layer(ins, affine=True)
        self.depthwise = torch.nn.ConvTranspose2d(ins, ins, kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=ins, bias=False)
        self.pointwise = torch.nn.Conv2d(ins, outs, 1, groups=groups, bias=False)

    def forward(self, ins):
        """ forward pass """
        return self.pointwise(self.norm(self.depthwise(ins)))


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
                 DWS=False,groups=1,dilation=1,norm_type='batch',
                 bias=False):
        assert kernel_size % 2 == 1
        super().__init__()
        padding_size = (kernel_size // 2)*dilation
        #self.pad = torch.nn.ReflectionPad2d(padding_size)
        self.pad = ReflectPad2d(padding_size)
        if padding == 'zero':
            self.pad = torch.nn.ZeroPad2d()(padding_size)
        self.conv2d = torch.nn.Conv2d(ins, outs, kernel_size, stride, 
                                      dilation=dilation,bias=bias)
        if DWS:
            self.conv2d = DWSConv(ins, outs, kernel_size, stride, 
                                  groups=groups,dilation=dilation,
                                  norm_type=norm_type,bias=bias)

    def forward(self, ins):
        """ forward pass """
        out = self.pad(ins)
        out = self.conv2d(out)
        return out


class Conv1stLayer(torch.nn.Module):
    """ Conv layer with reflection or zero padding """
    def __init__(self,ins,outs,kernel_size=3,stride=1,padding='ref',
                 DWS=False,groups=1,dilation=1,norm_type='batch'):
        assert kernel_size % 2 == 1
        super().__init__()
        padding_size = (kernel_size // 2)*dilation
        #self.pad = torch.nn.ReflectionPad2d(padding_size)
        self.DWS = DWS
        self.pad = ReflectPad2d(padding_size)
        if padding == 'zero':
            self.pad = torch.nn.ZeroPad2d()(padding_size)
        self.conv2d = torch.nn.Conv2d(ins, outs, kernel_size, stride, 
                                      dilation=dilation,bias=False)
        if DWS:
            self.conv2d = torch.nn.Conv2d(ins,outs,1,groups=1,bias=False)

    def forward(self, ins):
        """ forward pass """
        if self.DWS:
            out = ins
        else:
            out = self.pad(ins)
        out = self.conv2d(out)
        return out


class UpConv(torch.nn.Module):
    """ Up sample conv layer with zero padding
        note: conv transpose is used as this is compatible with coremltools
        worse for checkerboard artifacts - revisit with coreml version update
    """
    def __init__(self, ins, outs, kernel_size=4, upsample=2, padding='ref',
                 DWS=False, groups=1,norm_type='batch'):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(ins, outs, kernel_size, 
                                               stride=upsample, padding=1, 
                                               bias=False)
        if DWS:
            self.conv2d = DWSConvT(ins, outs, kernel_size, stride=upsample,
                                   padding=1, groups=groups,
                                   norm_type=norm_type)

    def forward(self, ins):
        """ forward pass """
        return self.conv2d(ins)


class UpConvUS(torch.nn.Module):
    """ Up sample conv layer with padding using upsampling """
    def __init__(self, ins, outs, kernel_size=3, upsample=2, padding='ref',
                 DWS=False, groups=1,norm_type='batch'):
        assert kernel_size % 2 == 1
        super().__init__()
        padding_size = kernel_size // 2
        self.pad = torch.nn.ReflectionPad2d(padding_size)
        #self.pad = ReflectPad2d_rev(padding_size)
        if padding == 'zero':
            self.pad = torch.nn.ZeroPad2d()(padding_size)
        self.conv2d = torch.nn.Conv2d(ins,outs,kernel_size,stride=1,bias=False)
        if DWS:
            self.conv2d = DWSConv(ins, outs, kernel_size,
                                  groups=groups,
                                  norm_type=norm_type)
        self.upsample = upsample

    def forward(self, x):
        out = x
        # upsample, then apply conv
        if self.upsample:
            # Note: float needed for scale in order to be compatible with jit
            out = torch.nn.functional.interpolate(out, mode='nearest', 
                                                  scale_factor=float(self.upsample))
        out = self.pad(out)
        out = self.conv2d(out)
        return out


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
        self.conv1 = Conv(channels,channels,kernel_size,DWS=DWS, groups=groups,
                          norm_type=norm_type)
        self.norm1 = norm_layer(channels, affine=True)
        self.leak = leak
        if leak == 0:
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            self.relu = torch.nn.LeakyReLU(leak)
        self.conv2 = Conv(channels,channels,kernel_size,DWS=DWS,
                          norm_type=norm_type)
        self.norm2 = norm_layer(channels, affine=True)

    def forward(self, ins):
        """ forward pass """
        res = ins
        out = self.relu(self.norm1(self.conv1(ins)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + res)


class Layer131(torch.nn.Module):
    """ 1-3-1 residual layer to import into ImageTransformer
        key component of shufflenetv2 and mobilenet v2
    """
    def __init__(self,ins,outs,mids,kernel_size=3,leak=0.05,norm_type='batch',
                 groups=1,dilation=1):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        padding_size = kernel_size // 2
        #self.pad = torch.nn.ReflectionPad2d(padding_size)
        self.pad = ReflectPad2d(padding_size)
        self.firstlayer = torch.nn.Conv2d(ins, mids, 1, 
                                          groups=groups, bias=False)
        self.depthwise = torch.nn.Conv2d(mids, mids, kernel_size, 
                                         groups=mids, bias=False,
                                         dilation=dilation)
        self.pointwise = torch.nn.Conv2d(mids, outs, 1, 
                                         groups=groups, bias=False)
        self.leak = leak
        if leak == 0:
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.relu2 = torch.nn.ReLU(inplace=True)
        else:
            self.relu1 = torch.nn.LeakyReLU(leak)
            self.relu2 = torch.nn.LeakyReLU(leak)
        self.norm1 = norm_layer(mids, affine=True)
        self.norm2 = norm_layer(mids, affine=True)
        self.norm3 = norm_layer(outs, affine=True)


    def forward(self, ins):
        """ forward pass """
        out = self.relu1(self.norm1(self.firstlayer(ins)))
        out = self.relu2(self.norm2(self.depthwise(self.pad(out))))
        return self.norm3(self.pointwise(out))


class ShuffleLayer(torch.nn.Module):
    """ Basic shuffle layer with 1-3-1 bottleneck layer """
    def __init__(self,channels,mids,kernel_size=3, 
                 leak=0.05,norm_type='batch',groups=1):
        super().__init__()
        self.main_branch = Layer131(channels,channels,channels // 2,
                                    kernel_size,leak=leak,
                                    norm_type=norm_type,groups=groups)

    def forward(self, old_x):
        x_proj, x = shuffle_v2(old_x)
        return torch.cat((x_proj, self.main_branch(x)), 1)


class ResShuffleLayer(torch.nn.Module):
    """ Basic residual layer to import into ImageTransformer
    """
    def __init__(self,channels,kernel_size=3,leak=0.05,norm_type='batch',
                 DWS=False, groups=1,dilation=1):
        super().__init__()
        if norm_type == 'batch':
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = torch.nn.InstanceNorm2d
        self.conv1 = Conv(channels,channels,kernel_size,DWS=DWS,groups=groups,
                          dilation=dilation,norm_type=norm_type)
        self.norm1 = norm_layer(channels, affine=True)
        self.leak = leak
        if leak == 0:
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            self.relu = torch.nn.LeakyReLU(leak)
        self.conv2 = Conv(channels,channels,kernel_size,DWS=DWS,groups=groups,
                          norm_type=norm_type)
        self.norm2 = norm_layer(channels, affine=True)
        self.groups = groups

    def forward(self, ins):
        """ forward pass """
        res = ins
        out = self.relu(self.norm1(self.conv1(ins)))
        out = self.norm2(self.conv2(out))
        return shuffle_v1(self.relu(out + res),self.groups)


class InvertedResidual(torch.nn.Module):
    """ MobileNetv2 style residual linear bottleneck layer
        to import into ImageTransformer
    """
    def __init__(self,ins,outs,expansion,stride=1,leak=0,dilation=1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.is_res = stride == 1 and ins == outs
        self.conv = Layer131(ins,outs,ins*expansion,kernel_size=3,
                             stride=stride,leak=leak,dilation=dilation)
        
    def forward(self, x):
        if self.is_res:
            return x + self.conv(x)
        else:
            return self.conv(x)
        