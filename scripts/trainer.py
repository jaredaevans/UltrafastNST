#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains both the stylization trainer and ae trainer 
    for the image transformer class 
"""

import torch

from vgg19 import build_vgg19, get_vgg19_content
from losses import GetContentLoss

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
        # get the content and content_style targets
        content, sty = self.run_content(inputs)

        # forward + backward + optimize
        self.vggmodel(self.transformer(inputs))
        
        style_score = 0
        content_score = 0
        cs_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for j, ct in enumerate(self.content_losses):
            content_score += self.cllayer(ct.value,content[j].to(self.device))
        if self.content_style_layers is not None:
            for j, ct in enumerate(self.cs_losses):
                cs_score += self.csllayer(ct.value,sty[j].to(self.device))
        var_score = self.var_loss.loss
        
        style_score *= self.style_weight
        content_score *= self.content_weight
        cs_score *= self.cs_weight
        var_score *= self.tv_weight
        
        loss = style_score + content_score + var_score
        # combine losses
        if self.content_style_layers is not None:
            loss += cs_score
        
        loss_tracker[0] += style_score.item()
        loss_tracker[1] += content_score.item()
        loss_tracker[2] += var_score.item()
        if self.content_style_layers is not None:
            loss_tracker[3] += cs_score.item()
        
        inputs = inputs.to(self.cpu)
        
        return loss                    
            
    
    def update_history(self, history, epoch, losses, k, val_losses=None, vk = None):
        """ record history information """
        print('epoch: {} - Style Loss: {:3f} Content Loss: {:3f} Var Loss: {:3f} CS Loss: {:3f}'.format(epoch + 1, losses[0]/k, losses[1]/k, losses[2]/k,losses[3]/k))
        history['total_loss'].append((losses[0]+losses[1]+losses[2]+losses[3])/k)
        history['style_loss'].append((losses[0])/k)
        history['content_loss'].append((losses[1])/k)
        history['var_loss'].append((losses[2])/k)
        if val_losses is not None:
            print('   Validation Style Loss: {:3f} Content Loss: {:3f} Var Loss: {:3f} CS Loss: {:3f}'.format(val_losses[0]/vk, val_losses[1]/vk, val_losses[2]/vk, val_losses[3]/vk))
            history['val_total_loss'].append((val_losses[0]+val_losses[1]+val_losses[2]+val_losses[3])/vk)
            history['val_style_loss'].append((val_losses[0])/vk)
            history['val_content_loss'].append((val_losses[1])/vk)
            history['val_var_loss'].append((val_losses[2])/vk)
            
            
    def train(self, data,  val=None, epochs = 1000, style_weight=10000,content_weight=1,
              tv_weight=100, cs_weight=10000, num_workers=0,batch_size=4, epoch_show = 20, 
              lr = 0.01, best_path = "best_run.pth", es_patience = 5):
        """ main training function """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.cs_weight = cs_weight
        
        # set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        
        history = {'total_loss': [],'style_loss': [],'content_loss': [],'var_loss': []}
        if val is not None:
            valloader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
            history = {'total_loss': [],'style_loss': [],'content_loss': [],
                       'var_loss': [],'val_total_loss': [],'val_style_loss': [],
                       'val_content_loss': [],'val_var_loss': []}
            es = 0
            vl_best = 1e8
        
        k = 0
        running_losses = [0,0,0,0]
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
                    val_losses = [0,0,0,0]
                    with torch.no_grad():
                        vk = 0
                        es += 1
                        for i, image_dat in enumerate(valloader, 0):
                            # validation step
                            inputs, content = image_dat
                            loss = self.step(inputs,val_losses)
                            vk = i
                        self.update_history(history, epoch, running_losses, k, val_losses, vk)
                        vl_cur = (val_losses[0] + val_losses[1] + val_losses[2] + val_losses[3])/vk             
                        if vl_cur < vl_best:
                            vl_best = vl_cur
                            torch.save(self.transformer.state_dict(), best_path)
                            es = 0
                        if es >= es_patience:
                            print("No improvement for {} epochs, ceasing training".format(es_patience))
                            break
                        
                else:
                    self.update_history(history, epoch, running_losses, k)
                running_losses = [0,0,0,0]
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