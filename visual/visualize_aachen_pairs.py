import sys
#sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import *
from top_imgs import top_images
from skimage.transform import resize
from skimage.exposure import match_histograms
from skimage.filters import sobel
from guidedbp_vanilla import pipe_line
from guidedbp_layer import pipe_line as pipeline_layer
from optm_visual import optm_visual
from torchvision import models
import configparser
import argparse
import pickle
import cv2
from scipy.io import loadmat
from PIL import ImageDraw
from PIL import ImageFont

colors = loadmat('logs/color150.mat')['colors']

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

layer = 4
block = 0

dataset = 'AachenPairs'
root_folder = osp.join('./figs',dataset+'_files')
mode_folder = osp.join(root_folder,'visualize_layer','layer_{}_block_{}'.format(layer,block))
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

# start guided bp/ vanilla
to_size = (224,224)
margin = 3

if not os.path.exists(osp.join(mode_folder,'checkpoint.txt')):
    checkpoint = 0
else:
    with open(osp.join(mode_folder,'checkpoint.txt'), 'r') as f:
            checkpoint = eval(f.read())
    checkpoint = checkpoint - 1

for idx,pair in enumerate(pairs[checkpoint:]):

    idx += checkpoint
    G_row = []
    for num_styles in [0,4,8,16]:
        
        print('Processing pair {}, style {}'.format(idx+1, num_styles))
        
        dataset_path = osp.join('data', dataset)
        day_path = pair[0]
        night_path = pair[1]

        assert 'day' in day_path, "Wrong day path"
        assert 'night' in night_path, "Wrong night path"
    
        grads_folder_day = osp.join(mode_folder,'day','grads_layer_{}'.format(layer))
        grads_folder_night = osp.join(mode_folder,'night','grads_layer_{}'.format(layer))
        if not os.path.exists(grads_folder_day):
            os.makedirs(grads_folder_day)
            os.makedirs(grads_folder_night)

        grads_path_day = osp.join(grads_folder_day,'grads_day_{}_layer_{}_block_{}_style_{}.txt'.format(idx+1,layer,block,num_styles))
        grads_path_night = osp.join(grads_folder_night,'grads_night_{}_layer_{}_block_{}_style_{}.txt'.format(idx+1,layer,block,num_styles))
        # load an image
        day = load_image(day_path)
        night = load_image(night_path)
        # preprocess an image, return a pytorch variable
        input_day = preprocess(day)
        input_night = preprocess(night)

        day = np.asarray(day.resize(to_size))
        night = np.asarray(night.resize(to_size))

        skip_col = np.uint8(np.ones((to_size[0],margin,3))*255)
        skip_row = np.uint8(np.ones((margin,2*to_size[0]+margin,3))*255)
        if num_styles == 0:
            G_img = np.concatenate((day,skip_col,night),axis=1)
            G_row.append(G_img)
            G_row.append(skip_row)
        # start visualization
        if not os.path.exists(grads_path_day):
            guided_grads_day = pipeline_layer(input_day, models[num_styles], layer, block,to_size)
            with open(grads_path_day,'wb') as f:
                pickle.dump(guided_grads_day,f)
        else:
            with open(grads_path_day,'rb') as f:
                guided_grads_day = pickle.load(f)
        
        if not os.path.exists(grads_path_night):
            guided_grads_night = pipeline_layer(input_night, models[num_styles], layer, block,to_size)
            with open(grads_path_night,'wb') as f:
                pickle.dump(guided_grads_night,f)
        else:
            with open(grads_path_night,'rb') as f:
                guided_grads_night = pickle.load(f)


        guided_grads_day = np.clip(guided_grads_day,0,guided_grads_day.max())
        guided_grads_night = np.clip(guided_grads_night,0,guided_grads_night.max())

        grads_output_day = norm_std(guided_grads_day.copy())
        grads_output_night = norm_std(guided_grads_night.copy())
        
        G_grads = np.concatenate((grads_output_day,skip_col,grads_output_night),axis=1)
        G_row.append(G_grads)
        if num_styles is not 16:
            G_row.append(skip_row)
    
    G_row = np.concatenate(G_row,axis=0)

    file_name_to_export = 'pair_'+str(idx+1)+'_layer_'+str(layer)+'_block_'+str(block)
    to_folder = osp.join(mode_folder)

    save_original_images(G_row, to_folder, file_name_to_export)

    
    with open(osp.join(mode_folder,'checkpoint.txt'), 'w') as f:
        print(idx+1, file=f)