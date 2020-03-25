"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
sys.path.append('../')
import cv2
import torch
from torch.nn import ReLU
from skimage.transform import resize
import numpy as np
from utils import *
import vecquantile as vecquantile
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,selected_layer,selected_block, to_size, filter_idx=None):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.to_size = to_size 
        self.num_maps = 1
        self.conv_output = 0
        self.gradients = None
        self.forward_relu_outputs = []
        self.strongest_filters = None
        self.filter_idx = filter_idx
        # Put model in evaluation mode
        self.model.eval()

        self.hook_list = []
        self.update_relus()
        self.hook_layer_backward()
        self.hook_layer_forward()

    
    def hook_layer_backward(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]

        self.hook_list.append(self.model._modules['conv1'].register_backward_hook(hook_function))

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            self.conv_output = output[0]
        # Hook the selected layer
        self.hook_list.append(self.model._modules['layer'+str(self.selected_layer)][self.selected_block]._modules['conv2'].register_forward_hook(hook_function))

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output 
            
            grad_relu = grad_in[0].clone()
            modified_grad_in =  torch.clamp(grad_relu, min=0.0)
           
            return (modified_grad_in,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
           
            self.forward_relu_outputs.append(ten_out)

        # hook up the first relu
      
        self.hook_list.append(self.model._modules['relu'].register_backward_hook(relu_backward_hook_function))
        self.hook_list.append(self.model._modules['relu'].register_forward_hook(relu_forward_hook_function))
        # Loop through layers, hook up ReLUs
        num_blocks = [3,4,6,3]
        for layer in range(1,4+1):
            for block in range(0,num_blocks[layer-1]):
              
                self.hook_list.append(self.model._modules['layer'+str(layer)][block]._modules['relu'].register_backward_hook(relu_backward_hook_function))
                self.hook_list.append(self.model._modules['layer'+str(layer)][block]._modules['relu'].register_forward_hook(relu_forward_hook_function))
                if layer == self.selected_layer and block == self.selected_block:
                    
                    return 
            

    def generate_gradients(self, input_image):

        # Forward pass
        torch.autograd.set_detect_anomaly(True)
        x = input_image
        _ = self.model(x)
        
        '''
        conv_np = self.conv_output.cpu().data.numpy()
        quantile = 0.01
        quant = vecquantile.QuantileVector(depth=self.conv_output.shape[0], seed=1)
        distr = np.transpose(conv_np, axes=(1, 2, 0)).reshape((-1, self.conv_output.shape[0]))
        quant.add(distr)
        thresholds = quant.readout(1000)[:, int(1000 * (1-quantile)-1)]
        mask_layer = 0
        for i in range(conv_np.shape[0]):
        
            mask = resize(conv_np[i], self.to_size)
            mask = 1.0*(mask > thresholds[i])
            mask_layer += mask
        mask_layer[mask_layer>0] = 1.0
        '''
        #print(mask_layer[mask_layer>0])
        # Zero gradients
        self.model.zero_grad()
        loss = torch.sum(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
        loss.backward()
        '''
        gradients_as_arr_list = []
        for i in range(self.conv_output.shape[0]):
            print("Processing Layer {}, Block {}, filter {}".format(self.selected_layer,self.selected_block,i))
            self.model.zero_grad()
            #loss = torch.sum(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
            loss = torch.max(self.conv_output[i])
            loss.backward(retain_graph=True)
            gradients_as_arr_list.append(resize(self.gradients.cpu().data.numpy()[0].transpose(1,2,0),self.to_size))
        #if self.selected_layer >= 3:
        #    idx = np.argsort(self.conv_output.detach().numpy().flatten())[::-1]
        #    print(idx[:5])
        #    print(self.conv_output.detach().numpy().flatten()[idx[:5]])
   
        '''
        
        gradients_as_arr = resize(self.gradients.cpu().data.numpy()[0].transpose(1,2,0),self.to_size)

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
       # gradients_as_arr = self.gradients.data.numpy()[0]

        #threshold_scale = 0.5
        #mask_all = np.zeros((224,224))
        #for feature_map in self.conv_output:
        #    feature_map = feature_map.data.numpy()
        #    feature_map = feature_map / np.max(feature_map)
       
         #   mask = cv2.resize(feature_map, (224,224))
         #   mask[mask < 0.5*np.max(feature_map)] = 0.0 # binarize the mask
         #   mask[mask > 0.5*np.max(feature_map)] = 1.0
         #   mask_all += mask
        #mask_all[mask_all>1.0] = 1.0

        # unhook for the next run to avoid getting a model every time
        for hook in self.hook_list:
            hook.remove()

        return gradients_as_arr#,mask_layer
    
    def get_strongest_filters(self, activation_img, top=1):
    
        activation_img = activation_img.detach().numpy()
        # Find maximum activation for each filter for a given image
        activation_img = np.nanmax(activation_img, axis=3)
        activation_img = np.nanmax(activation_img, axis=2)

        activation_img = activation_img.sum(0)

        # Make activations 1-based indexing
        #activation_img = np.insert(activation_img, 0, 0.0)

        #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
        #  each entry corresponds to the maximum/summed activation of each filter for a given image

        top_filters = activation_img.argsort()[-top:]
        return list(top_filters)

def pipe_line(img, model, layer, block, to_size, filter_idx = None):
    
   
    input_img = img
    input_img.requires_grad = True
    
    #device = torch.cuda.device(0)
    #model = model.cuda()
    input_img = input_img.cuda()
    # Guided backprop
    GBP = GuidedBackprop(model, layer, block, to_size, filter_idx = filter_idx)
    # Get gradients
    guided_grads = GBP.generate_gradients(input_img)


    return guided_grads


