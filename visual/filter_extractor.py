import sys
sys.path.append('../')
import os
import os.path as osp
import numpy as np
from utils import *

import pickle
import torch

class StrongFilterExtractor():
    def __init__(self,model,image_tensor):
        self.model = model
        self.model.eval()
        self.images = image_tensor
        self.filter_maxima = {}
        self.init_hooks(self.model._modules)
    def get_strong_filters(self,layer,block_idx):
        def hook(module,input,output):
            conv_output = output.data.numpy()
            strong_activations = np.max(conv_output,axis=(2,3))
            strong_filter_indices = np.argmax(strong_activations,axis=1)
            strong_filter_values = np.amax(strong_activations, axis=1)
            self.filter_maxima[layer]['block'+str(block_idx)] = (strong_filter_values, strong_filter_indices)
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

def generate_strong_filters(model,dataset,path1,path2):
    path = osp.join('data', dataset)
    img_dirs = os.listdir(path) 
    img_tensor = torch.zeros(len(img_dirs),3,224,224)
    for i,name in enumerate(img_dirs):
        img_path = osp.join(path,name)
        img = load_image(img_path)
        img = preprocess(img)
        img_tensor[i] = img
    

    filterExtractor = StrongFilterExtractor(model, img_tensor)
    filterMaxima = filterExtractor.get_maxima()
    torch.save(filterMaxima,path1)

    with open(path2, 'wb') as fp:
        pickle.dump(img_dirs, fp)

if __name__ == "__main__":
    style = 16
    weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
    weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[style])
    model = get_model(weights_dir)
    model.eval()
    filter_maxima_path = '../0311_filter'+'style_'+str(style)+'.pt'
    img_dirs_path = '../0311_img'+'style_'+str(style)+'.txt'
    generate_strong_filters(model,'AachenDay',filter_maxima_path,img_dirs_path)