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
#from guidedbp_layer import pipe_line as pipeline_layer
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
'''
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
'''
SG_list = []
param_n = 25
param_sigma_multiplier = 3
to_size = (112,112)

for name in weights_names:
    weights_dir = osp.join('../scripts/logs/stylized_models',name)
    SG_list.append(SmoothGradients(get_model(weights_dir).cuda().eval(),4,2,param_n, param_sigma_multiplier, mode = 'GBP',to_size=to_size))
# define root folder
dataset = 'AachenNight'
root_folder = osp.join('./figs',dataset+'_files')
mode_folder = osp.join(root_folder,'visualize_layer_stats')
if not os.path.exists(mode_folder):
    os.makedirs(mode_folder)
# img directories
path = osp.join('data', dataset)
if not os.path.exists(osp.join(mode_folder,'img_dirs.txt')):  
    img_dirs = os.listdir(path)
    with open(osp.join(mode_folder,'img_dirs.txt'), 'w') as f:
        print(img_dirs, file=f)
else:
    with open(osp.join(mode_folder,'img_dirs.txt'), 'r') as f:
        img_dirs = eval(f.read())

# start guided bp/ vanilla
to_size = (112,112)
margin = 3
num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):

        stats_path = osp.join(mode_folder,'layer_{}_block_{}'.format(layer,block)+'_stats.txt')
        if not os.path.exists(stats_path):
            stats_dict = {}
            for num_styles in [0,4,8,16]:
                stats_dict[num_styles] = {}
                for label in ['building','sky','light','others']:
                    stats_dict[num_styles][label] = []
        else:
            with open(stats_path, 'r') as f:
                    stats_dict = eval(f.read())

        checkpoint = len(stats_dict[16]['others'])-1 if len(stats_dict[0]['others']) is not 0 else 0

        for idx,img_dir in enumerate(img_dirs[checkpoint:]):
            idx += checkpoint
            grads_normld_list = []
            label_list = []
            for i,num_styles in enumerate([0,4,8,16]):
            
                print('Processing image {}/{}, style {}'.format(idx,len(img_dirs)-1, num_styles))
                
                dataset_path = osp.join('data', dataset)
                img_path = osp.join(dataset_path,img_dir)
                mask_path = osp.join(root_folder,'binary_masks',img_dir.split('.')[0]+'_mask.png')
                seg_path = osp.join(root_folder,'semantic_masks',img_dir.split('.')[0]+'.txt')
            
                grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
                if not os.path.exists(grads_folder):
                    os.makedirs(grads_folder)
                grads_path = osp.join(grads_folder,'grads_img_{}_layer_{}_block_{}_style_{}.txt'.format(idx,layer,block,num_styles))
                # load an image
                img = load_image(img_path)
                # preprocess an image, return a pytorch variable
                input_img = preprocess(img)

                mask = cv2.imread(mask_path, 0).astype(np.uint8)
                mask = np.repeat(mask[:,to_size[1]:,None],3,axis=2)
            
                label = np.loadtxt(seg_path,dtype=np.uint8,delimiter=',')

                building_sky = np.zeros_like(label)
                building_sky[label==2] = 1
                building_sky = sobel(building_sky)

                # assign a color index
                label[building_sky!=0] = 67

                img_label = colorEncode(label, colors).astype(np.uint8)

                img_label[mask==255] = mask[mask==255]

                label[mask[:,:,0]==255] = 255

                #img_label = colorEncode(label, colors).astype(np.uint8)
                #img_label[mask==255] = mask[mask==255]
                label[mask[:,:,0]==255] = 255
                label_list.append(label)

                if not os.path.exists(grads_path):
                    guided_grads = SG_list[i].pipe_line(input_img.cuda())
                    with open(grads_path,'wb') as f:
                        pickle.dump(guided_grads,f)
                else:
                    with open(grads_path,'rb') as f:
                        guided_grads = pickle.load(f)

                guided_grads = np.clip(guided_grads,0,guided_grads.max())
                grads_normld_list.append((guided_grads - guided_grads.min())/(guided_grads.max()-guided_grads.min()))


                
            factor = 1#np.sum(np.stack(grads_normld_list))
            for i,num_styles in enumerate([0,4,8,16]):
                label = label_list[i]
                grads_normld = grads_normld_list[i]
                grads_normld_summed = np.sum(grads_normld,axis=2)

                building_index = (label==1) + (label==0) + (label==6)
                building_grad = np.sum(grads_normld_summed[building_index])/len(np.nonzero(building_index)[0]) \
                                                            if len(np.nonzero(building_index)[0]) is not 0 else 0
                #building_grad_ratio = building_grad/len(np.nonzero(building_index)[0]) #/ np.sum(grads_normld_summed)

                light_index = label==255
                light_grad = np.sum(grads_normld_summed[light_index])/len(np.nonzero(light_index)[0]) \
                                                             if len(np.nonzero(light_index)[0]) is not 0 else 0
                #light_grad_ratio = light_grad / len(np.nonzero(light_index)[0])#np.sum(grads_normld_summed)


                near_sky_index = label==67
                near_sky_grad = np.sum(grads_normld_summed[near_sky_index]) / len(np.nonzero(near_sky_index)[0]) \
                                                    if len(np.nonzero(near_sky_index)[0]) is not 0 else 0
                #near_sky_grad_ratio = near_sky_grad / len(np.nonzero(near_sky_index)[0])#np.sum(grads_normld_summed)

                others_index = (np.ones_like(near_sky_index) - (building_index+light_index+near_sky_index)*1) == 1
                others_grad = np.sum(grads_normld_summed[others_index]) / len(np.nonzero(others_index)[0])
                #others_grad_ratio = others_grad / len(np.nonzero(near_sky_index)[0])#np.sum(grads_normld_summed)

                if len(stats_dict[num_styles]['building'])>= idx+1:
                    pass
                else:
                    stats_dict[num_styles]['building'].append(building_grad)

                if len(stats_dict[num_styles]['light'])>= idx+1:
                    pass
                else:
                    stats_dict[num_styles]['light'].append(light_grad)
                
                if len(stats_dict[num_styles]['sky'])>= idx+1:
                    pass
                else:
                    stats_dict[num_styles]['sky'].append(near_sky_grad)
                
                if len(stats_dict[num_styles]['others'])>= idx+1:
                    pass
                else:
                    stats_dict[num_styles]['others'].append(others_grad)
        

            
                with open(stats_path, 'w') as f:
                    print(stats_dict, file=f)

