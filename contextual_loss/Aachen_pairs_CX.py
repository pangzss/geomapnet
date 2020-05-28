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

from CX_distance_ori import CX_loss

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
dataset = 'AachenPairs'
root_folder = osp.join('./figs',dataset+'_files')
mode_folder = osp.join(root_folder,'contextual_loss')
if not os.path.exists(mode_folder):
    os.makedirs(mode_folder)
# img directories
path = osp.join('data', dataset)
if not os.path.exists(osp.join(mode_folder,'pairs.txt')):  
    dirs = os.listdir(path)

    pairs = []
    for i in range(len(dirs)):
        pair_path = osp.join(path,'pair{}'.format(i+1))
        pair = os.listdir(pair_path)
        day = osp.join(pair_path, pair[0] if 'day' in pair[0] else pair[1])
        night = osp.join(pair_path,pair[0] if 'night' in pair[0] else pair[1])

        pairs.append((day,night))
    with open(osp.join(mode_folder,'pairs.txt'), 'w') as f:
        print(pairs, file=f)
else:
    with open(osp.join(mode_folder,'pairs.txt'), 'r') as f:
        pairs = eval(f.read())



# hook layers
layer = 4
block = 0

cx_loss_dict = {}
cx_loss_dict[0] = []
cx_loss_dict[4] = []
cx_loss_dict[8] = []
cx_loss_dict[16] = []
for idx,pair in enumerate(pairs):

    print('Processing pair {}/{}'.format(idx+1,len(pairs)))
    dataset_path = osp.join('data', dataset)
    day_path = pair[0]
    night_path = pair[1]

    assert 'day' in day_path, "Wrong day path"
    assert 'night' in night_path, "Wrong night path"

    # load an image
    day = load_image(day_path)
    night = load_image(night_path)
    # preprocess an image, return a pytorch variable
    input_day = preprocess(day)
    input_night = preprocess(night)

    input_day = input_day.cuda()
    input_night = input_night.cuda()

    cx_collector = []
    for num_styles in [0,4,8,16]:
        model = models[num_styles]
        feats_go = get_feats(model,layer,block)
        _ = model(input_day)
        day_feats = feats_go.feats.clone()
        _ = model(input_night)
        night_feats = feats_go.feats.clone()

        cx_loss = CX_loss(day_feats, night_feats)
        cx_loss = cx_loss.item()
        if not np.isnan(cx_loss):
            cx_collector.append(cx_loss)

    if len(cx_collector) == 4:
        for i,num_styles in enumerate([0,4,8,16]):
            cx_loss_dict[num_styles].append(cx_collector[i])

        #feats_go.hook.remove()

with open(osp.join(mode_folder,'layer_{}_block_{}_cx_loss.txt'.format(layer,block)), 'w') as f:
    print(cx_loss_dict, file=f)
