import sys
sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from common.utils import *

import cv2

from PIL import ImageDraw
from PIL import ImageFont

from CX_distance import CX_loss

features = {}
# hook
class get_feats():
    def __init__(self,model,layer,block):
        self.model = model
        self.layer = layer
        self.block = block
        self.feats = None
        self.hook = None
        self.hook_layer()
    def hook_layer(self):
        def hook_function(module, input, output):
            self.feats = output

        self.hook = self.model._modules['layer'+str(self.layer)][self.block]._modules['conv2'].register_forward_hook(hook_function)
        

# get model
weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[0])
model_0 = get_model(weights_dir)
model_0.cuda()
model_0.eval()

weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[4])
model_4 = get_model(weights_dir)
model_4.cuda()
model_4.eval()

weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[8])
model_8 = get_model(weights_dir)
model_8.cuda()
model_8.eval()

weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[16])
model_16 = get_model(weights_dir)
model_16.cuda()
model_16.eval()

models = {0:model_0,
          4:model_4,
          8:model_8,
          16:model_16}

# define root folder
dataset = 'AachenPairsRaw'
root_folder = osp.join('./figs','AachenPairs_files')
mode_folder = osp.join(root_folder,'perceptual_loss_rand_pairs')
if not os.path.exists(mode_folder):
    os.makedirs(mode_folder)
# img directories
path = osp.join('data', dataset,'all')
if not os.path.exists(osp.join(mode_folder,'img_dirs.txt')):  

    img_dirs = os.listdir(path)
    with open(osp.join(mode_folder,'img_dirs.txt'), 'w') as f:
        print(img_dirs, file=f)
else:
    with open(osp.join(mode_folder,'img_dirs.txt'), 'r') as f:
        img_dirs = eval(f.read())



# hook layers
num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        pc_loss_dict = {}
        pc_loss_dict[0] = []
        pc_loss_dict[4] = []
        pc_loss_dict[8] = []
        pc_loss_dict[16] = []
        for idx in range(116):

            print('Processing pair {}/{}'.format(idx+1,116))
            dataset_path = osp.join('data', dataset,'all')
            pair_idces = np.random.choice(len(img_dirs),2)
            img1_path = osp.join(dataset_path,img_dirs[pair_idces[0]])
            img2_path = osp.join(dataset_path,img_dirs[pair_idces[1]])
            # load an image
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            # preprocess an image, return a pytorch variable
            input_img1 = preprocess(img1) 
            input_img2 = preprocess(img2)

            input_img1 = input_img1.cuda()
            input_img2 = input_img2.cuda()

            for num_styles in [0,4,8,16]:
                model = models[num_styles]
                feats_go = get_feats(model,layer,block)
                _ = model(input_img1)
                img1_feats = feats_go.feats.clone()[0].cpu().data.numpy()
                _ = model(input_img2)
                img2_feats = feats_go.feats.clone()[0].cpu().data.numpy()


                C,H,W = img1_feats.shape

                img1_feats = np.reshape(img1_feats,(C,H*W))
                img2_feats = np.reshape(img2_feats,(C,H*W))

                img1_feats_sq = np.sum(img1_feats**2,axis=1)
                img2_feats_sq = np.sum(img2_feats**2,axis=1)
                product = np.sum(img1_feats*img2_feats,axis=1)

                perceptual_loss = np.sum((img1_feats_sq - 2*product + img2_feats_sq)/(C*H*W))
                pc_loss_dict[num_styles].append(perceptual_loss)


                #feats_go.hook.remove()

        with open(osp.join(mode_folder,'layer_{}_block_{}_pc_loss.txt'.format(layer,block)), 'w') as f:
            print(pc_loss_dict, file=f)
