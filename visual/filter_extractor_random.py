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
        self.filter_maxima = {}
        self.init_hooks(self.model._modules)
    def get_strong_filters(self,layer,block_idx):
        def hook(module,input,output):
            conv_output = output.data.numpy()
            featmap_tag = np.zeros((conv_output.shape[0],conv_output.shape[1],2))
            # generate tags for each feature map to indicate which image it belongs to (last_dim[0]) and which filter it comes from (last_dim[1])
            for i in range(featmap_tag.shape[0]):
                featmap_tag[i,:,0] = int(i)*np.ones(featmap_tag.shape[1],dtype=int)
                for j in range(featmap_tag.shape[1]):
                    featmap_tag[i,j,1] = int(j)  
            conv_output = np.reshape(conv_output,(-1,conv_output.shape[2],conv_output.shape[3]))
            featmap_tag = np.reshape(featmap_tag,(-1,2))

            rand_indices = np.random.permutation(conv_output.shape[0])[:conv_output.shape[0]//100]

            conv_output_sub = conv_output[rand_indices]
            featmap_tag_sub = featmap_tag[rand_indices]
            if self.criterion == 'max':
                strong_activations = np.max(conv_output_sub,axis=(1,2))
            elif self.criterion == 'sum':
                #alpha = 0.7
                #strong_activations = (1-alpha)*np.max(conv_output_sub,axis=(1,2))+alpha*np.mean(conv_output_sub,axis=(1,2))
                strong_activations = np.sum(conv_output_sub,axis=(1,2))
            else:
                KeyError
            
            self.filter_maxima[layer]['block'+str(block_idx)] = (strong_activations, featmap_tag_sub)
        return hook
    # registering hooks for the conv resnet
    def init_hooks(self,modules):
        for layer_str in modules:
            if 'layer' in layer_str :
                self.filter_maxima[layer_str] = {}
        
                for block_idx,block in enumerate(modules[layer_str]):
                    block.register_forward_hook(self.get_strong_filters(layer_str,block_idx))

    def get_maxima(self):
        '''this function only takes resnet34 module'''
        
        _ = self.model(self.images)

 
        return self.filter_maxima

def generate_strong_filters(model,dataset,path1,path2, iterations=None,criterion='max'):
    path = osp.join('data', dataset)
    img_dirs = os.listdir(path)
    img_tensor = torch.zeros(len(img_dirs),3,224,224)
    for i,name in enumerate(img_dirs):
        img_path = osp.join(path,name)
        img = load_image(img_path)
        img = preprocess(img)
        img_tensor[i] = img
    

    if iterations == None:
        filterExtractor = StrongFilterExtractor(model, img_tensor)
        filterMaxima = filterExtractor.get_maxima()
    else:
        filterMaxima = []
        for i in range(iterations):
            filterExtractor = StrongFilterExtractor(model, img_tensor,criterion=criterion)
            filterMaxima.append(filterExtractor.get_maxima())
    torch.save(filterMaxima,path1)

    with open(path2, 'wb') as fp:
        pickle.dump(img_dirs, fp)

