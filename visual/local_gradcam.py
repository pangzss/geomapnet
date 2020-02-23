import os
import sys
sys.path.append('../')

import torch
from torch.nn import ReLU
import matplotlib.pyplot as plt

import torchvision.models as models

from PIL import Image
import numpy as np
from utils import *
from common.train import load_state_dict
from models.posenet import PoseNet, MapNet
from conv_resnet import resnet34
import configparser

class CamExtractor():
    """
        Extracts cam features from the model (mapnet)
    """
    def __init__(self, model, target_layer, target_block):
        self.model = model
        self.target_layer = target_layer
        self.target_block = target_block
        self.gradients = None
        self.conv_output = None
        self.hook_list = []
        self.hook_layer()
    def save_gradient(self,grad): # hook for tensor
        
        self.gradients = grad
    def hook_layer(self):
        def hook_function(module,input,output):
            self.hook_list.append(output.register_hook(self.save_gradient))
            self.conv_output = output
        try:
            resnet = self.model._modules['mapnet']._modules['feature_extractor']
            self.hook_list.append(resnet._modules['layer'+str(self.target_layer)][self.target_block]._modules['conv2'].register_forward_hook(hook_function))
        except KeyError:
            self.hook_list.append(self.model._modules['layer'+str(self.target_layer)][self.target_block]._modules['conv2'].register_forward_hook(hook_function))
        
    def forward(self,x):
        
        return self.model(x)

class GradCam():
    def __init__(self, model, target_layer,target_block):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer,target_block)
    def generate_cam(self, input_image):
        
        model_output = self.extractor.forward(input_image)
        conv_output = self.extractor.conv_output
        self.model.zero_grad()
        try:
            
            dLdp = torch.ones(1,3,2, dtype=torch.float)
            model_output.backward(gradient=dLdp)
        except RuntimeError:
            
            dLdp = torch.FloatTensor(1, 1000).zero_()
            max_idx = torch.argmax(model_output)
            
            dLdp[0][max_idx] = 1
            model_output.backward(gradient=dLdp)

        dpdx = self.extractor.gradients.data.numpy()[0]
        target = conv_output.data.numpy()[0]
        weights = np.mean(dpdx,axis=(1,2))
    
        cam = np.ones(target.shape[1:],dtype=np.float32)
        for i,w in enumerate(weights):
        
            cam += w * target[i,:,:]
        print(cam.max())
        cam = np.maximum(cam,0)
       
        #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = cam -np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(cam*255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        
        for hook in self.extractor.hook_list:
            hook.remove()

        return cam

def get_model(task,pretrained):
    if task == 'classification':
        model = models.resnet34(pretrained=pretrained)
        
    else:
        # adapted resnet34 with forward hooks
        settings = configparser.ConfigParser()
        with open('../scripts/configs/style.ini','r') as f:
            settings.read_file(f)
        section = settings['hyperparameters']
        dropout = section.getfloat('dropout')

        feature_extractor = models.resnet34(pretrained=False)
        posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)

        mapnet_model = MapNet(mapnet=posenet)
        if pretrained == True:
            # load weights
            loc_func = lambda storage, loc: storage
            #weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
            weights_dir = './logs/stylized_models/AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar'
            checkpoint= torch.load(weights_dir,map_location=loc_func)
            load_state_dict(mapnet_model,checkpoint['model_state_dict'])

        #feature_extractor = mapnet_model._modules['mapnet']._modules['feature_extractor']
        model = mapnet_model
    return model

def pipe_line(model, img_path, layer, block, pretrained=False,task='classification'):
  
    img_file = img_path
    # load an image
    img = load_image(img_file)
    
    # preprocess an image, return a pytorch variable
    input_img,ori_image = preprocess_cam(img)
    #input_img.requires_grad = True
   
    # Guided backprop
    grad_cam = GradCam(model, layer, block)
    # Get gradients
    cam = grad_cam.generate_cam(input_img)
    # Save colored gradients
    
    file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)
    
    save_grad_cam(ori_image, cam, file_name_to_export, pretrained,task)
    #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    #plt.imshow(grayscale_guided_grads[0],cmap='gray')
    #plt.imshow(pos_sal.transpose(1,2,0))
    #plt.show(block=False)
    #plt.show()
    print('Grad cam completed. Layer {}, block {}'.format(layer, block))

if __name__ == '__main__':
 
    task_list = ['classification','localization']
    img_paths = ['./imgs/cat_dog.png',
                 './imgs/aachen1.jpg']
    task_index = 0
    task = task_list[task_index]
    img_path = img_paths[task_index]

    pretrained = True
    model = get_model(task,pretrained)
    num_blocks = [3,4,6,3]

    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):
            
            pipe_line(model, img_path, layer, block, pretrained=pretrained,task=task)
   
