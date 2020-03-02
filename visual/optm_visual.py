"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
import os.path as osp
sys.path.append('../')

import torch
from torch.nn import ReLU
from torch.optim import Adam
import numpy as np
from utils import *

class optm_visual():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,selected_layer,selected_block, filter_idx):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.filter_idx = filter_idx
        
        self.conv_output = None
        # Put model in evaluation mode
        self.model.eval()
        self.hook = None
        self.hook_layer_forward()

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output[0, self.filter_idx]
        # Hook the selected layer
        self.hook = self.model._modules['layer'+str(self.selected_layer)][self.selected_block].register_forward_hook(hook_function)

   
    def generate_optm_visual(self):
         # Generate a random image
        random_image = np.uint8(np.random.normal(150, 180, (224, 224, 3)))
        visual_image = preprocess_image(random_image, False)
        visual_image.requires_grad = True
        optimizer = Adam([visual_image], lr=0.1, weight_decay=1e-6)
        for i in range(1,201):
            optimizer.zero_grad()
            x = visual_image
            out = self.model(x)

            loss = -torch.mean(self.conv_output)
            if loss == 0:
                loss += 1

            if i % 5 == 0:
                print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))

            loss.backward()
            optimizer.step()

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        self.hook.remove()

        visual_image = recreate_image(visual_image)
        return visual_image

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


if __name__ == '__main__':
    layer = 4
    block = 0
    filter_idx = 29
    style = 0
    weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
    weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[style])
    model = get_model(weights_dir)
    #print(model._modules['layer4'][1]._modules['conv2'].weight.data[0,filter_idx])
    visual_image = optm_visual(model,layer,block, filter_idx).generate_optm_visual()

    folder = './figs/optm/style_'+str(style)
    if not os.path.exists(folder):
            os.makedirs(folder)
    name = 'layer_'+str(layer)+'_block_'+str(block)+'_filter_'+str(filter_idx)+'.png'
    path = osp.join(folder,name)
    save_image(visual_image,path)