#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Dataset definitions for loading datasets
"""

import torch


class ContentDataset(torch.utils.data.Dataset):
    """ join the x and y into a dataset"""
    def __init__(self, imgs, content):
        """
        Args:
            imgs(tensor): loaded x dataset (output from above)
            content(tensor): content output from  trainer.get_content_targets(imgs)
        """
        self.imgs = imgs
        self.content = content

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.imgs[idx], self.content[idx]]
        return sample


class ContentandStyleDataset(torch.utils.data.Dataset):
    """ join the x and y into a dataset"""
    def __init__(self, imgs, content, style):
        """
        Args:
            imgs(tensor): loaded x dataset (output from above)
            content(tensor): content output from  trainer.get_content_targets(imgs)
            style(tensor): style output from content image (used for color)
                from trainer.get_content_targets(imgs, style_too = style_layers)
        """
        self.imgs = imgs
        self.content = content
        self.style = style

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.imgs[idx], self.content[idx], self.style[idx]]
        return sample


class InputDataset(torch.utils.data.Dataset):
    """ join the x into a dataset - used for training as an AE """
    def __init__(self, imgs):
        """
        Args:
            imgs(tensor): loaded x image dataset
        """
        self.imgs = imgs
        self.nulltens = torch.Tensor([0])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.imgs[idx], self.nulltens]
        return sample


class PortraitSegDataset(torch.utils.data.Dataset):
    """ join the x, y into a dataset """
    def __init__(self, imgs, masks):
        """
        Args:
            imgs(tensor): loaded x image dataset
        """
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.imgs[idx], self.masks[idx]]
        return sample
