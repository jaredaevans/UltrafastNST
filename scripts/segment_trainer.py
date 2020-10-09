#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains both the stylization trainer and ae trainer 
    for the image transformer class 
"""

import torch
import matplotlib.pyplot as plt

from losses import SoftDiceLoss, FocalLoss, JSloss


def prepandclip(img):
    return img.squeeze().data.clamp_(-1, 1).cpu().detach()


def show_test_image_quality(model, image, device):
    """
    Parameters
    ----------
    model : image transformer
    image : test image in PIL -> torch.tensor format
    device : cpu or GPU
    Returns
    -------
    Prints original image beside stylized image
    """
    model_input = image.clone() 
    image = (image.squeeze(0).to(torch.device("cpu")).permute(1,2,0)+1.)/2
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    plt.title('input')
    # stylize
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)
    output = prepandclip(model_output)
    output = (output.to(torch.device("cpu")).permute(1,2,0)+1.)/2
    # styled plot
    plt.subplot(122)
    plt.imshow(output)
    plt.axis('off')
    plt.title('output')
    # show figure
    plt.tight_layout()
    plt.show()


class SegmentTrainer():
    """ Segmenter trainer """
    def __init__(self,segmenter):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmenter = segmenter.to(device)
        if device == torch.device("cuda"):
            self.segmenter.cuda()
        else:
            self.segmenter.cpu()
        self.optimizer = torch.optim.Adam(self.segmenter.parameters(),
                                          lr=0.01,betas=(0.9,0.999))
        self.diceloss = SoftDiceLoss()
        self.focalloss = FocalLoss(gamma=2)
        self.KLloss = FocalLoss(gamma=1)
        self.JSloss = JSloss()
        self.CEloss = FocalLoss(gamma=0)
        
    def calcIOU(self, mask, mask_pred):
        """ use torch tensors with values of 0 or 1 """
        mask_pred_x = mask_pred.clone().detach()
        thresh = 1.64872
        mask_pred_x[mask_pred_x<=thresh] = 0
        mask_pred_x[mask_pred_x>thresh] = 1
        
        sum1 = mask + mask_pred_x
        sum1[sum1>0] = 1
        sum1[sum1<=0] = 0
        x = torch.sum(sum1)
        if x == 0: # union = 0
            return 1.
        sum2 = mask + mask_pred_x
        sum2[sum2<2] = 0
        sum2[sum2>=2] = 1
        y = torch.sum(sum2)
        return 1.0*(y/x).item()
        
    def step(self,imgs,masks,edge,losses):
        """ take a step on gpu """
        
        # put on gpu
        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        edge = edge.to(self.device)
        
        # add noise to image for stabilization
        #noise = torch.zeros_like(imgs)
        #noise.normal_(mean=0, std=0.016)
        # get the content and content_style targets

        # forward + backward + optimize
        masks_pred, edge_pred = self.segmenter(imgs)
        
        # IOU is not used for loss, but is for tracking
        IoU = self.calcIOU(masks.unsqueeze(1), masks_pred)
        
        # mask losses
        if self.mask_loss == "KL":
            mask_score = self.KLloss(masks_pred,masks.unsqueeze(1)*1.)
        elif self.mask_loss == "softKL":
            mask_score = self.KLloss(masks_pred,masks.unsqueeze(1)*0.9+0.05)
        elif self.mask_loss == "JS":
            mask_score = self.JSloss(masks_pred,masks.unsqueeze(1)*1.)
        elif self.mask_loss == "softJS":
            mask_score = self.JSloss(masks_pred,masks.unsqueeze(1)*0.9+0.05)
        elif self.mask_loss == "CE":
            mask_score = self.CEloss(masks_pred,masks.unsqueeze(1)*1.)
        elif self.mask_loss == "dice":
            mask_score = self.diceloss(masks_pred,masks.unsqueeze(1))
        elif self.mask_loss == "focal":
            mask_score = self.focalloss(masks_pred,masks.unsqueeze(1))
        else:
            if not self.complained_mask:
                print("Invalid mask loss selected; defaulting to KL")
                self.complained_mask = True
            mask_score = self.KLloss(masks_pred,masks.unsqueeze(1)*1.)
        
        # Edge Losses
        if self.edge_loss == "KL":
            edge_score = self.KLloss(edge_pred,edge.unsqueeze(1)*1.)
        elif self.edge_loss == "softKL":
            edge_score = self.KLloss(edge_pred,edge.unsqueeze(1)*0.9+0.05)
        elif self.edge_loss == "JS":
            edge_score = self.JSloss(edge_pred,edge.unsqueeze(1)*1.)
        elif self.edge_loss == "softJS":
            edge_score = self.JSloss(edge_pred,edge.unsqueeze(1)*0.9+0.05)
        elif self.edge_loss == "CE":
            edge_score = self.CEloss(edge_pred,edge.unsqueeze(1)*1.)
        elif self.edge_loss == "dice":
            edge_score = self.diceloss(edge_pred,edge.unsqueeze(1))
        elif self.edge_loss == "focal":
            edge_score = self.focalloss(edge_pred,edge.unsqueeze(1))
        else:
            if not self.complained_edge and self.edge_loss is not None:
                print("Invalid edge loss selected; defaulting to None")
                self.complained_edge = True
            edge_score = torch.tensor(0)

        mask_score *= self.mask_weight
        
        loss = mask_score + edge_score
        
        losses[0] += mask_score.item()
        losses[1] += edge_score.item()
        losses[2] += IoU
        
        imgs = imgs.to(self.cpu)
        masks = masks.to(self.cpu)
        edge = edge.to(self.cpu)
        
        return loss
    
    def train(self,data,val=None,epochs=10,lr=0.01,batch_size=16,num_workers=1,
              epoch_show=20,best_path="best.pth",es_patience=5,mask_weight=10,
              test_image=None,test_im_show=5,
              mask_loss='JS',edge_loss='JS'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.mask_weight = mask_weight
        #set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.mask_loss = mask_loss
        self.edge_loss = edge_loss
        self.complained_mask = False
        self.complained_edge = False
            
        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        
        if val is not None:
            valloader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
            es = 0
            vl_best = 1e8
        
        for epoch in range(epochs): 
            k = 0
            running_losses = [0,0,0,0,0]
            print("On epoch {} of {}".format(epoch+1,epochs))
            # loop over the dataset multiple times
            for i, image_dat in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, content]
                imgs_ori, imgs, edges, masks = image_dat

                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                loss = self.step(imgs, masks, edges, running_losses)
                loss.backward()
                self.optimizer.step()
                k += 1
            num_ev = (k+1)*batch_size
            print("  Training on {} images;  IoU = {}; Mask loss = {}; Edge loss = {}".format(num_ev,running_losses[2]/(k+1),running_losses[0]/(k+1),running_losses[1]/(k+1)))
            if val is not None:
                val_losses = [0,0,0,0,0,0]
                with torch.no_grad():
                    vk = 0
                    es += 1
                    for i, image_dat in enumerate(valloader, 0):
                        # validation step
                        imgs_ori, imgs, edges, masks = image_dat
                        loss = self.step(imgs, masks, edges, val_losses)
                        vk = i
                    num_ev = (vk+1)*batch_size
                    print("  Validation on {} images:  IoU = {}; Mask loss = {}; Edge loss = {}".format(num_ev,val_losses[2]/(vk+1),val_losses[0]/(vk+1),val_losses[1]/(vk+1)))
                    #self.update_history(history,epoch,running_losses,k,
                    #                    val_losses, vk)
                    vl_cur = sum(val_losses)/vk             
                    if vl_cur < vl_best:
                        vl_best = vl_cur
                        torch.save(self.segmenter.state_dict(), best_path)
                        es = 0
                    if test_image is not None:
                        if epoch % test_im_show == test_im_show - 1:
                            show_test_image_quality(self.segmenter,
                                                    test_image,self.device)
                    if es >= es_patience:
                        print("No improvement for {} epochs, ceasing training".format(es_patience))
                        break
    