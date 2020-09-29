#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains both the stylization trainer and ae trainer 
    for the image transformer class 
"""

import torch
import random
import matplotlib.pyplot as plt

from vgg19 import build_vgg19, get_vgg19_content
from losses import GetContentLoss


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


class Trainer():
    """ Stylization trainer """
    def __init__(self,transformer,content_layers,style_layers,style_image,
                 content_style_layers = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = transformer.to(device)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_style_layers = content_style_layers
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),lr=0.01,
                                          betas=(0.98,0.9999))
        self.vggmodel, self.style_losses, self.content_losses, self.var_loss, self.cs_losses \
                = build_vgg19(style_image,style_layers,content_layers,
                              content_style_layers=content_style_layers)
        self.cllayer = GetContentLoss()
        if self.content_style_layers is not None:
            self.csllayer = GetContentLoss()
        self.stabilitylayer = GetContentLoss()
                
        
    def get_content_targets(self, content_imgs):
        # build out a content target through the network
        content, style = get_vgg19_content(content_imgs,self.content_layers,
                                    self.content_style_layers)
        return content, style
    
    def run_content(self, batch):
        """ Pass content images, skip tranformer and get targets for this batch 
            (avoids preloading in memory)
        """
        content, style = [], []
        with torch.no_grad():
            self.vggmodel(batch)
            for j, ct in enumerate(self.content_losses):
                content.append(ct.value)
            if self.content_style_layers is not None:
                for j, ct in enumerate(self.cs_losses):
                    style.append(ct.value)
        return content, style

    def step(self,inputs,loss_tracker):
        """ take a step on gpu """
        # put on gpu
        inputs = inputs.to(self.device)
        
        # add noise to image for stabilization
        noise = torch.zeros_like(inputs)
        noise.normal_(mean=0, std=0.016)
        # get the content and content_style targets
        content, sty = self.run_content(inputs+noise)

        # forward + backward + optimize
        stylized_image = self.transformer(inputs+noise)
        self.vggmodel(stylized_image)
        
        style_score = 0
        content_score = 0
        cs_score = 0
        
        # style loss
        for sl in self.style_losses:
            style_score += sl.loss
        # content loss
        for j, ct in enumerate(self.content_losses):
            content_score += self.cllayer(ct.value,content[j].to(self.device))
        # content_style loss
        if self.content_style_layers is not None:
            for j, ct in enumerate(self.cs_losses):
                cs_score += self.csllayer(ct.value,sty[j].to(self.device))
        # variational loss
        var_score = self.var_loss.loss
        # stability loss (inject noise, shift image, require similiar results)
        dx = random.choice([-4,-3,-2,-1,1,2,3,4])
        dy = random.choice([-4,-3,-2,-1,1,2,3,4])
        inputs_shift = torch.roll(inputs, shifts=(dx, dy), dims=(2, 3))
        noise.normal_(mean=0, std=0.016)
        sty_image_shift = self.transformer(inputs_shift + noise)
        sty_image_shift = torch.roll(sty_image_shift,
                                     shifts=(-dx, -dy),dims=(2, 3))
        stability_score = self.stabilitylayer(sty_image_shift[:,:,5:-5,5:-5],
                                              stylized_image[:,:,5:-5,5:-5]) 
        
        
        style_score *= self.style_weight
        content_score *= self.content_weight
        cs_score *= self.cs_weight
        var_score *= self.tv_weight
        stability_score *= self.stability_weight
        
        loss = style_score + content_score + var_score + stability_score
        # combine losses
        if self.content_style_layers is not None:
            loss += cs_score
        
        loss_tracker[0] += style_score.item()
        loss_tracker[1] += content_score.item()
        loss_tracker[2] += var_score.item()
        loss_tracker[3] += stability_score.item()
        if self.content_style_layers is not None:
            loss_tracker[4] += cs_score.item()
        
        inputs = inputs.to(self.cpu)
        
        return loss                    
            
    
    def update_history(self, history, epoch, losses, k, val_losses=None, vk = None):
        """ record history information """
        print('epoch: {} - Losses:: Style: {:.3g} Content: {:.3g} Var: {:.3g} Stable: {:.3g} CS: {:.3g}'.format(epoch + 1, losses[0]/k, losses[1]/k, losses[2]/k,losses[3]/k,losses[4]/k))
        history['total_loss'].append(sum(losses)/k)
        history['style_loss'].append((losses[0])/k)
        history['content_loss'].append((losses[1])/k)
        history['var_loss'].append((losses[2])/k)
        history['stable_loss'].append((losses[3])/k)
        if val_losses is not None:
            print('   Validation Losses:: Style: {:.3g} Content: {:.3g} Var: {:.3g} Stable: {:.3g} CS: {:.3g}'.format(val_losses[0]/vk, val_losses[1]/vk, val_losses[2]/vk, val_losses[3]/vk, val_losses[4]/vk))
            history['val_total_loss'].append(sum(val_losses)/vk)
            history['val_style_loss'].append((val_losses[0])/vk)
            history['val_content_loss'].append((val_losses[1])/vk)
            history['val_var_loss'].append((val_losses[2])/vk)
            history['val_stable_loss'].append((val_losses[3])/vk)
            
            
    def train(self,data,val=None,epochs = 1000,style_weight=10000,
              content_weight=1,tv_weight=100, cs_weight=10000, stable_weight=1,
              num_workers=0,batch_size=4,epoch_show = 20,lr=0.01,
              best_path="best.pth",es_patience=5,test_image=None,test_im_show=5):
        """ main training function """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.cs_weight = cs_weight
        self.stability_weight = stable_weight
        
        # set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        
        history = {'total_loss': [],'style_loss': [],'content_loss': [],
                   'var_loss': [],'stable_loss': []}
        if val is not None:
            valloader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
            history = {'total_loss': [],'style_loss': [],'content_loss': [],
                       'var_loss': [],'stable_loss': [],'val_total_loss': [],
                       'val_style_loss': [],'val_content_loss': [],
                       'val_var_loss': [],'val_stable_loss': []}
            es = 0
            vl_best = 1e8
        
        k = 0
        running_losses = [0,0,0,0,0]
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, image_dat in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, content]
                inputs, content = image_dat

                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                loss = self.step(inputs,running_losses)
                loss.backward()
                self.optimizer.step()
                k += 1
            # print statistics
            if epoch % epoch_show == epoch_show - 1:
                if val is not None:
                    val_losses = [0,0,0,0,0]
                    with torch.no_grad():
                        vk = 0
                        es += 1
                        for i, image_dat in enumerate(valloader, 0):
                            # validation step
                            inputs, content = image_dat
                            loss = self.step(inputs,val_losses)
                            vk = i
                        self.update_history(history,epoch,running_losses,k,
                                            val_losses, vk)
                        vl_cur = sum(val_losses)/vk             
                        if vl_cur < vl_best:
                            vl_best = vl_cur
                            torch.save(self.transformer.state_dict(), best_path)
                            es = 0
                        if test_image is not None:
                            if epoch % test_im_show == test_im_show - 1:
                                show_test_image_quality(self.transformer,
                                                        test_image,self.device)
                        if es >= es_patience:
                            print("No improvement for {} epochs, ceasing training".format(es_patience))
                            break
                        
                else:
                    self.update_history(history, epoch, running_losses, k)
                running_losses = [0,0,0,0,0]
                k = 0
                
        print("Training complete")  
        return history


class IdentityTrainer():
    """ AE trainer """
    def __init__(self, transformer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = transformer.to(device)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=0.015,betas=(0.98,0.9999))

    def train_to_identity(self, data, val=None, epochs = 1000,num_workers=0,batch_size=4, 
                          epoch_show = 1, best_path = "best.pth", lr = 0.01):
        """ 
        trains the network to pass images through with minimal loss
        """
        """ main training function """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cpu = torch.device("cpu")
        
        # set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        history = {'loss': []}
        if val is not None:
            valloader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
            history = {'loss': [], 'valloss': []}
        
        cllayer = torch.nn.L1Loss()
        
        self.transformer.to(device)
        
        running_loss = 0
        k = 0
        vl_best = 1e5
        for epoch in range(epochs):  # loop over the dataset multiple time
            for i, image_dat in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, content]
                inputs, content = image_dat
                inputs = inputs.to(device)
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.transformer(inputs)
                loss = cllayer(inputs,outputs) 
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                k+=1
                inputs = inputs.to(cpu)
                
            # print statistics
            if epoch % epoch_show == epoch_show - 1:
                if val is not None:
                    val_loss = 0
                    with torch.no_grad():
                        vk = 0
                        for i, image_dat in enumerate(valloader, 0):
                            inputs, content = image_dat
                            inputs = inputs.to(device)
                            outputs = self.transformer(inputs)
                            loss = cllayer(inputs,outputs) 
                            val_loss += loss.item()
                            vk = i
                            inputs = inputs.to(cpu)
                        vl_cur = val_loss/vk
                        print('epoch: {} - Loss: {:3f} - ValLoss: {:3f}'.format(epoch + 1, running_loss/k, vl_cur))
                        history['loss'].append(running_loss/k)
                        history['valloss'].append(vl_cur)
                        if vl_cur < vl_best:
                            vl_best = vl_cur
                            torch.save(self.transformer.state_dict(), best_path)
                else:
                    print('epoch: {} - Loss: {:3f}'.format(epoch + 1, running_loss/k))
                    history['loss'].append(running_loss/k)
                running_loss = 0
                k = 0
                
        print("Training complete")        
        return history