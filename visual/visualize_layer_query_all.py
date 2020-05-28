import sys
# sys.path.append('../')
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
# from guidedbp_layer import pipe_line as pipeline_layer
from smooth_grads import SmoothGradients
from optm_visual import optm_visual
from torchvision import models
import configparser
import argparse
import pickle
import cv2
from scipy.io import loadmat
from PIL import ImageDraw
from PIL import ImageFont

colors = loadmat('../../logs/color150.mat')['colors']

# get model
weights_names = ['AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar']
SG_list = []
param_n = 25
param_sigma_multiplier = 3
to_size = (224,224)

for name in weights_names:
    weights_dir = osp.join('../scripts/logs/stylized_models', name)
    SG_list.append(
        SmoothGradients(get_model(weights_dir).cuda().eval(), 4, 2, param_n, param_sigma_multiplier, mode='GBP',
                        to_size=to_size))
# define root folder
dataset = 'AachenNight'
root_folder = osp.join('./figs', dataset + '_files')
mode_folder = osp.join(root_folder, 'visualize_layer_query_all')
if not os.path.exists(mode_folder):
    os.makedirs(mode_folder)
# img directories
path = osp.join('data', dataset)
img_dirs = os.listdir(path)

# start guided bp/ vanilla
margin = 3
num_blocks = [3, 4, 6, 3]

layer = 4
block = 2


for idx, img_dir in enumerate(img_dirs):

    G_col = []
    for i, num_styles in enumerate([0, 4, 8, 16]):

        print('Processing image {}/{}, style {}'.format(idx, len(img_dirs) - 1, num_styles))

        dataset_path = osp.join('data', dataset)
        img_path = osp.join(dataset_path, img_dir)

        grads_folder = osp.join(mode_folder, 'grads_layer_{}'.format(layer))
        if not os.path.exists(grads_folder):
            os.makedirs(grads_folder)
        grads_path = osp.join(grads_folder,
                              'grads_img_{}_layer_{}_block_{}_style_{}.txt'.format(idx, layer, block,
                                                                                   num_styles))
        # load an image
        img = load_image(img_path)
        img = img.resize(to_size)
        #ori_img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
        if i == 0:
            # discrep_col.append(img)
            # discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
            G_col.append(img)
            G_col.append(np.uint8(np.ones((to_size[0], margin, 3)) * 255))
        # preprocess an image, return a pytorch variable
        input_img = preprocess(img,to_size)

        if not os.path.exists(grads_path):
            guided_grads = SG_list[i].pipe_line(input_img.cuda())
            with open(grads_path, 'wb') as f:
                pickle.dump(guided_grads, f)
        else:
            with open(grads_path, 'rb') as f:
                guided_grads = pickle.load(f)

        guided_grads = np.clip(guided_grads, 0, guided_grads.max())
        grads_output = norm_std(guided_grads.copy())
        G_col.append(grads_output)
        G_col.append(np.uint8(np.ones((to_size[0], margin, 3)) * 255))
    G_col = np.concatenate(G_col, axis=1)
    file_name_to_export = 'layer_' + str(layer) + '_block_' + str(block) + '_data_' + str(idx)
    to_folder = osp.join(mode_folder)
    save_original_images(G_col[:, :-margin], to_folder, file_name_to_export)