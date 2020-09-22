#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Helper module that handles the use of VGG19 in the training of the model
"""

import torch
import torch.nn as nn
import torchvision.models as models

from losses import VariationalLoss, ContentTrack, StyleLoss, StyleTrack, gram_matrix


class Normalization(nn.Module):
    """ Layer to normalize input image so we can easily put it in the
    network
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        """ forward pass """
        # normalize img
        return (img - self.mean) / self.std


def build_vgg19(style_img,
                style_layers,
                content_layers,
                content_style_layers=None):
    """ Master function for building the vgg19 to be placed at the
        end of the image transformer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load VGG properties
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn.training = False

    # Use normalization to convert [-1,1] images
    # norm for [0,1] are mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # converstion is mean' = 2*mean-1 and std' = 2*std
    cnn_normalization_mean = torch.tensor([-0.03, -0.088, -0.188]).to(device)
    cnn_normalization_std = torch.tensor([0.458, 0.448, 0.45]).to(device)

    # normalization module
    normalization = Normalization(cnn_normalization_mean,
                                  cnn_normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    # NOTE: content loss needs a) pre-evaluation of content on training set
    #       and b) method to grab yout and use in content evaluation
    content_losses = []
    style_losses = []
    content_style_losses = []

    model = nn.Sequential()

    #add variational loss layer for smooth images
    vl = VariationalLoss()
    model.add_module("var_loss", vl)
    var_loss = vl

    # add normalization for VGG format
    model.add_module("norm", normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # inplace = False needed to keep from modifying
            # earlier layers
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            content_loss = ContentTrack()
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            #print("{} {} {}".format(i,name,target_feature.shape))

        if content_style_layers is not None:
            if name in content_style_layers:
                # add optional content image style loss
                c_s_loss = StyleTrack()
                model.add_module("content_style_loss_{}".format(i), c_s_loss)
                content_style_losses.append(c_s_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentTrack, StyleLoss, StyleTrack)):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses, var_loss, content_style_losses


def get_vgg19_content(content_imgs, content_layers, content_style_layers=None):
    """ This only needs to be run once to get the y_values """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    # load VGG properties
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn.training = False
    # Use normalization to convert [-1,1] images
    # norm for [0,1] are mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # converstion is mean' = 2*mean-1 and std' = 2*std
    cnn_normalization_mean = torch.tensor([-0.03, -0.088, -0.188]).to(device)
    cnn_normalization_std = torch.tensor([0.458, 0.448, 0.45]).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses.  Note: we simply build the content style vals for simplicity
    # even if content_style_layers=None.  There is no impact on memory
    content_vals = []
    content_style_vals = []
    for _ in content_imgs:
        content_vals.append([])
        content_style_vals.append([])

    # normalization module
    normalization = Normalization(cnn_normalization_mean,
                                  cnn_normalization_std).to(device)

    # add normalization for VGG format
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # inplace = False needed to keep from modifying earlier layers
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            for j, content in enumerate(content_imgs):
                target = model(content.unsqueeze(0)).detach()
                content_vals[j].append(target.squeeze(0).to(cpu))

        if content_style_layers is not None:
            if name in content_style_layers:
                # add content loss:
                for j, content in enumerate(content_imgs):
                    target = model(content.unsqueeze(0)).detach()
                    content_style_vals[j].append(
                        gram_matrix(target).squeeze(0).to(cpu))

    return content_vals, content_style_vals
