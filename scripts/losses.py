#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Loss functions and loss layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    """ gram matrix for feature assignments """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a feature map (N=c*d)
    allG = []
    for i in range(a):
        features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL
        gram = torch.mm(features, features.t())  # compute the gram product
        gram = gram.div(c * d)  # 'normalize' the values of the gram matrix
        allG.append(gram)
    return torch.stack(allG)


class StyleLoss(nn.Module):
    """ Style loss layer for target style image """
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        """ forward pass """
        gram = gram_matrix(input)
        batch_size = gram.shape[0]
        #self.loss = F.mse_loss(
        self.loss = F.l1_loss(
            gram.squeeze(),
            self.target.repeat(batch_size, 1, 1, 1).squeeze())
        return input


class ContentTrack(nn.Module):
    """ This modules tracks the content loss across many images
        for loading (partnered with next module)
    """
    def forward(self, input):
        """ forward pass """
        self.value = input.clone()
        return input


class GetContentLoss(nn.Module):
    """ Gets the content mse loss
    """
    def forward(self, input, target):
        """ forward pass """
        return F.mse_loss(input, target)


class StyleTrack(nn.Module):
    """ This modules tracks the content image style across many images
        for loading (partnered with next module). This is useful for
        e.g. maintaining color scheme of the content images
    """
    def forward(self, input):
        """ forward pass """
        gram = gram_matrix(input)
        self.value = gram
        return input


class GetStyleLoss(nn.Module):
    """ evaluate the style loss with gram matrix """
    def forward(self, input, target):
        """ forward pass """
        gram = gram_matrix(target)
        return F.mse_loss(gram, input)


class VariationalLoss(nn.Module):
    """ Variational loss to enforce continuity of images
    """
    def forward(self, input):
        """ forward pass """
        self.loss = F.mse_loss(input[:, :, 1:, :],
                               input[:, :, :-1, :]) + F.mse_loss(
                                   input[:, :, :, 1:], input[:, :, :, :-1])
        return input
