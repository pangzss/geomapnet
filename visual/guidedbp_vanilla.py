"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
sys.path.append('../')

import torch
from torch.nn import ReLU

import numpy as np
from utils import *

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,selected_layer,selected_block, method, filter_idx=None):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.method = method 
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
            # Gets the conv output of the selected filter (from selected layer)
            if self.filter_idx == None:
                activation_maps = output
                strong_idces = self.get_strongest_filters(activation_maps, top=self.num_maps)
                
                self.conv_output = output[0, strong_idces]
            else:
                self.conv_output = output[0, self.filter_idx]
        # Hook the selected layer
        self.hook_list.append(self.model._modules['layer'+str(self.selected_layer)][self.selected_block].register_forward_hook(hook_function))

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
   
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            if self.method == 'guidedbp':
                modified_grad_in =  torch.clamp(grad_in[0], min=0.0)
            elif self.method == 'vanilla':
                modified_grad_in =  grad_in[0] # vanilla
            del self.forward_relu_outputs[-1]  # Remove last forward output
           
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
        x = input_image
        x = self.model(x)
        
        # Zero gradients
        self.model.zero_grad()
        loss = torch.max(self.conv_output)
        loss.backward()


        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]

        # unhook for the next run to avoid getting a model every time
        for hook in self.hook_list:
            hook.remove()

        return gradients_as_arr
    
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

def pipe_line(img, model, layer, block, method, filter_idx = None):
    
    assert method == 'guidedbp' or method == 'vanilla', KeyError
    input_img = img.clone()
    input_img.requires_grad = True
    
    
    # Guided backprop
    GBP = GuidedBackprop(model, layer, block, method, filter_idx = filter_idx)
    # Get gradients
    guided_grads = GBP.generate_gradients(input_img)
   
    return guided_grads


