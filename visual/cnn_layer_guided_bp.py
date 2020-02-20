"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
sys.path.append('../')

import torch
from torch.nn import ReLU
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import *


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,selected_layer,selected_block):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.num_maps = 9
        self.conv_output = 0
        self.gradients = None
        self.forward_relu_outputs = []
        self.strongest_filters = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layer_backward()
        self.hook_layer_forward()

         # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')
    def hook_layer_backward(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]

        self.model._modules['conv1'].register_backward_hook(hook_function)

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            activation_maps = output
            strong_idces = self.get_strongest_filters(activation_maps, top=self.num_maps)
            
            self.conv_output = output[0, strong_idces]
        # Hook the selected layer
        self.model._modules['layer'+str(self.selected_layer)][self.selected_block].register_forward_hook(hook_function)

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
            print(grad_in[0].shape,grad_out[0].shape)
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
           # modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            modified_grad_out =  torch.clamp(grad_in[0], min=0.0)

            del self.forward_relu_outputs[-1]  # Remove last forward output
           
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
           
            self.forward_relu_outputs.append(ten_out)

        # hook up the first relu
        self.model._modules['relu'].register_backward_hook(relu_backward_hook_function)
        self.model._modules['relu'].register_forward_hook(relu_forward_hook_function)
        # Loop through layers, hook up ReLUs
        num_blocks = [3,4,6,3]
        for layer in range(1,4+1):
            for block in range(0,num_blocks[layer-1]):
              
                self.model._modules['layer'+str(layer)][block]._modules['relu'].register_backward_hook(relu_backward_hook_function)
                self.model._modules['layer'+str(layer)][block]._modules['relu'].register_forward_hook(relu_forward_hook_function)
                if layer == self.selected_layer and block == self.selected_block:
                    
                    return 
            

    def generate_gradients(self, input_image):
        # Forward pass
        


        x = input_image
        x = self.model(x)
        
        # Zero gradients
        self.model.zero_grad()
        loss = torch.mean(self.conv_output)
        loss.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def get_strongest_filters(self, activation_img, top=3):
    
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

def preprocess(img):
    img = np.asarray(img.resize((224, 224))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)
    print(img.shape)
    img = img[None,:,:,:]
    return img

def load_image(filename):
    img = Image.open(filename)
    return img
    
def get_patch_slice(actv_map,max_index,kernel_size=3):
    H,W = actv_map.shape[2:]
    h,w = max_index[2:]

    radius = (kernel_size - 1)/2

    left = w - radius
    right = w + radius + 1
    w_slice = slice( int(left*(left>0)), int(right - (right-W)*(right>W)))

    up = h - radius
    down = h + radius + 1
    h_slice = slice(int(up*(up>0)), int(down - (down-H)*(down>H)))

    return (max_index[0],max_index[1],h_slice, w_slice)

def get_strongest_filters(activation_img, top=3):
    
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


if __name__ == '__main__':

    img_file = './imgs/cat_dog.png'
    # load an image
    img = load_image(img_file)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img)
    input_img.requires_grad = True

    layer = 4
    block = 2
  
    pretrained_model = models.resnet34(pretrained=True)
    # Guided backprop
    GBP = GuidedBackprop(pretrained_model, layer, block)
    # Get gradients
    guided_grads = GBP.generate_gradients(input_img)
    # Save colored gradients
    #save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    #grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    #save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    #save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

    plt.imshow(pos_sal.transpose(1,2,0))
    plt.show(block=False)
    plt.show()
    print('Guided backprop completed')
