import sys
sys.path.append('../')
import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from utils import *
from common.train import load_state_dict
from models.posenet import PoseNet, MapNet
import pickle


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

def get_model(task,pretrained):
    if task == 'classification':
        model = models.resnet34(pretrained=pretrained)
    else:
        # adapted resnet34 with forward hooks
        feature_extractor = models.resnet34(pretrained=False)
        posenet = PoseNet(feature_extractor, droprate=0., pretrained=False)

        mapnet_model = MapNet(mapnet=posenet)
        if pretrained == True:
            # load weights
            loc_func = lambda storage, loc: storage
            #weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
            weights_dir = './logs/stylized_models/AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar'
            checkpoint= torch.load(weights_dir,map_location=loc_func)
            load_state_dict(mapnet_model,checkpoint['model_state_dict'])

        feature_extractor = mapnet_model._modules['mapnet']._modules['feature_extractor']
        model = feature_extractor 
    return model


if __name__ == '__main__':
    
    dataset = 'AachenDay'
    path = osp.join('data', dataset)
    img_dirs = os.listdir(path) 
    img_tensor = torch.zeros(len(img_dirs),3,224,224)
    for i,name in enumerate(img_dirs):
        img_path = osp.join(path,name)
        img = load_image(img_path)
        img = preprocess(img)
        img_tensor[i] = img
    
    pretrained = True
    task = 'localization'
    model = get_model(task,pretrained)
    

    filterExtractor = StrongFilterExtractor(model, img_tensor)
    filterMaxima = filterExtractor.get_maxima()
    torch.save(filterMaxima,'./AachenDay_files/filterMaxima.pt')

    with open('./AachenDay_files/img_dirs.txt', 'wb') as fp:
        pickle.dump(img_dirs, fp)