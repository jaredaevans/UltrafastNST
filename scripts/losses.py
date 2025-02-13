#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Loss functions and loss layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import unique

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

def luminous_val(input, color_bins, vec_dot):
    """ weighted sum of all pixels """
    a, b, c, d = input.size()
    weight = b*c*d
    allL = []
    for i in range(a):
        pix_sum = input.sum()/weight
        allL.append(pix_sum)
    return torch.stack(allL)

def color_bin(input, color_bins, bin_scale, n_tot):
    """ weighted sum of all pixels """
    batch_size, channels, height, width = input.size()
    all_col = []
    for i in range(batch_size):
        bucketed = torch.bucketize(input[i],color_bins)
        bins = torch.einsum('bcd, b -> cd', bucketed, bin_scale).to(torch.float)
        col_hist = torch.histc(bins,n_tot,min=0,max=n_tot-1)/(height*width)
        all_col.append(col_hist)
    return torch.stack(all_col)


class ColorLoss(nn.Module):
    """ color loss uses a discretized binning over N**3 bins
        then applys an MAE on this to evaluate
        the similarity of the two samples
    """
    def __init__(self, style_image, n_bins, device):
        super().__init__()
        self.device = device
        self.cpu = torch.device("cpu")
        self.n_bins = n_bins
        self.n_tot = n_bins ** 3
        self.bin_scale = torch.tensor([1, n_bins, n_bins**2]).to(self.cpu)
        ranging = 2.0/n_bins
        binning_array = []
        for i in range(1,n_bins):
            binning_array.append(i*ranging)
        self.color_bins = (torch.tensor(binning_array) - 1.).to(self.cpu)
        self.target = color_bin(style_image.to(self.cpu),self.color_bins,
                                self.bin_scale, self.n_tot).detach().squeeze()

    def forward(self, input):
        """ forward pass """
        input.to(self.cpu)
        cols = color_bin(input,self.color_bins,self.bin_scale, self.n_tot)
        batch_size = cols.shape[0]
        self.loss = F.l1_loss(
            cols.squeeze(),
            self.target.repeat(batch_size, 1).squeeze())
        input.to(self.device)
        return input


class StyleLoss(nn.Module):
    """ Style loss layer for target style image """
    def __init__(self, target_feature, weights=1):  
        super().__init__()
        if type(weights) is tuple:
            self.weights = weights
        else:
            self.weights = (0,weights)
        self.target = gram_matrix(target_feature).detach()
        
    def new_weights(self, weights):
        if type(weights) is tuple:
            self.weights = weights
        else:
            self.weights = (0,weights)

    def forward(self, input):
        """ forward pass """
        gram = gram_matrix(input)
        batch_size = gram.shape[0]
        #self.loss = F.mse_loss(
        targ = self.target.repeat(batch_size, 1, 1, 1).squeeze()
        l1_loss = F.l1_loss(gram.squeeze(), targ) 
        l2_loss = F.mse_loss(gram.squeeze(), targ) 
        self.loss = self.weights[0] * l1_loss + self.weights[1] * l2_loss 
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




"""  The following code is for the portrait segmenter:
        - SoftDiceLoss: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
        - FocalLoss: https://github.com/CoinCheung/pytorch-loss/
        - JSLoss: new
"""

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
    
    
""" focal loss """
class FocalSigmoidLossFunc(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLoss(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    '''
    def __init__(self,
                 alpha=1,
                 gamma=2,
                 reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFunc.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class JSloss(nn.Module):
    """  Compute the Jensen-Shannon loss using the torch native kl_div"""
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.red = reduction
        
    def forward(self, input, target):
         net = (input + target)/2.
         return 0.5 * (F.kl_div(input, net, reduction=self.red) + 
                       F.kl_div(target, net, reduction=self.red))
