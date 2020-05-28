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
class IntegratedGradients():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,selected_layer,selected_block, to_size=None):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.to_size = to_size 
        self.conv_output = 0
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()

        self.hook_layer_forward()
        self.hook_layer_backward()
    
    def hook_layer_backward(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]  

        self.model._modules['conv1'].register_backward_hook(hook_function)

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            self.conv_output = output[0]
        # Hook the selected layer
        self.model._modules['layer'+str(self.selected_layer)][self.selected_block].register_forward_hook(hook_function)
    
    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list


    def generate_gradients(self, input_image):

        # Forward pass
        _ = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        loss = torch.sum(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
        #loss = torch.max(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
        #loss = self.conv_output[0].view(-1)[0]
        #loss = torch.sum(self.conv_output)
        loss.backward()
        return self.gradients

    def generate_integrated_gradients(self, input_image,steps=10):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = torch.zeros_like(input_image)
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = resize(integrated_grads.cpu().data.numpy()[0].transpose(1,2,0),self.to_size)

        return gradients_as_arr    

    def pipe_line(self,img):
        
        input_img = img
        input_img.requires_grad = True
        
        # Get gradients
        integr_grads = self.generate_integrated_gradients(input_img.cuda())


        return integr_grads


if __name__ == "__main__":
    inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

    model = models.resnet34(pretrained=True).cuda()
    #weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'

    #model = get_model(weights_dir).cuda()
    img_name = 'snake.jpg'
    img_path = os.path.join('imgs',img_name)
    img = load_image(img_path)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img)
    ori_img =  inv_normalize(input_img[0]).cpu().numpy().transpose(1,2,0)

    IG = IntegratedGradients(model,4,2,(224,224))
    IG_grads = IG.pipe_line(input_img)
    #IG_grads = np.clip(IG_grads,0,IG_grads.max())

    grads_output = convert_to_grayscale(IG_grads)
    #grads_output = ori_img * grads_output.repeat(3,axis=2)
    file_name_to_export = '1_IG_'+img_name.split('.')[0]
    to_folder = 'imgs'

    save_original_images(ori_img, to_folder, 'resized_'+img_name)
    save_original_images(grads_output, to_folder, file_name_to_export)

