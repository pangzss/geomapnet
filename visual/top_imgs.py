import sys
sys.path.append('../')
import os
import os.path as osp
import numpy as np
from utils import *

import pickle
import torch

class StrongFilterExtractor():
    def __init__(self,model,image_tensor,criterion='max'):
        self.model = model
        self.model.eval()
        self.images = image_tensor
        self.criterion = criterion
        self.summed_activations = {}
        self.init_hooks(self.model._modules)
    def get_image_activations(self,layer,block_idx):
        def hook(module,input,output):

            conv_output = output.data.numpy()
            summed_activations = conv_output.sum(axis=(3,2,1))
            
            self.summed_activations[layer]['block'+str(block_idx)] = summed_activations
        return hook
    # registering hooks for the conv resnet
    def init_hooks(self,modules):
        for layer_str in modules:
            if 'layer' in layer_str :
                self.summed_activations[layer_str] = {}
        
                for block_idx,block in enumerate(modules[layer_str]):
                    block.register_forward_hook(self.get_image_activations(layer_str,block_idx))

    def get_maxima(self):
        '''this function only takes resnet34 module'''
        
        _ = self.model(self.images)

 
        return self.summed_activations

def top_images(model,dataset,topk=8):
    path = osp.join('data', dataset)
    img_dirs = os.listdir(path)
    img_tensor = torch.zeros(len(img_dirs),3,224,224)
    for i,name in enumerate(img_dirs):
        img_path = osp.join(path,name)
        img = load_image(img_path)
        img = preprocess(img)
        img_tensor[i] = img
    

    
    filterExtractor = StrongFilterExtractor(model, img_tensor)
    img_activations = filterExtractor.get_maxima()
    
    num_blocks = [3,4,6,3]
    top_img_dirs = {}
    for layer in range(1,4+1):
        top_img_dirs['layer'+str(layer)] = {}
        for block in range(0,num_blocks[layer-1]):
            activations = img_activations['layer'+str(layer)]['block'+str(block)]
            sorted_indices = np.argsort(activations)[::-1][:topk]
            top_img_dirs['layer'+str(layer)]['block'+str(block)] = [img_dirs[int(i)] for i in sorted_indices]
    
    return top_img_dirs