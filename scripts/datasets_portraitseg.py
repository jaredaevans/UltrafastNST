#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
By Jared Evans  

Modified from https://github.com/dong-x16/PortraitNet
"""
import torch

import os
import cv2
import numpy as np
import random

from PIL import Image

from data_augments import data_aug_blur, data_aug_color, data_aug_noise
from data_augments import data_aug_flip, aug_matrix
from data_augments import show_edge, Normalize_Img

class PortraitSegDatasetAug(torch.utils.data.Dataset):
    """ join the x, y into a dataset """
    def __init__(self, imgs, masks):
        """
        Args:
            imgs(tensor): loaded x image dataset
        """
        self.imgs = imgs
        self.masks = masks
        self.input_width = 64
        self.input_height = 64

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = None
        mask = None
        bbox = None
        H = None
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.imgs[idx]
        mask = self.masks[idx]

        height, width, channel = img.shape
        bbox = [0, 0, width-1, height-1]
        H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                   angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
        
        img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(self.padding_color, self.padding_color, self.padding_color)) 
        mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.input_width, self.input_height), 
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
        
        # add augmentation
        img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))  
        img_aug = data_aug_color(img_aug)
        img_aug = np.asarray(img_aug)
        # img_aug = data_aug_light(img_aug)
        img_aug = data_aug_blur(img_aug)
        img_aug = data_aug_noise(img_aug)
        img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread
        
        input_norm = Normalize_Img(img_aug, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
        input_ori_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)

        # put channels first
        input = np.transpose(input_norm, (2, 0, 1))
        input_ori = np.transpose(input_ori_norm, (2, 0, 1))
        
        output_mask = cv2.resize(mask_aug_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
        output_mask[output_mask>=0.5] = 1
        output_mask[output_mask<0.5] = 0
        edge = show_edge(output_mask)
        # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
        return input_ori, input, edge, output_mask

