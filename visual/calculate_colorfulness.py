import sys
#sys.path.append('../')
import os
import os.path as osp

import numpy as np

from utils import *
import pickle
import cv2
import matplotlib.pyplot as plt

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

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

colness = {}

num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        
        stats_path = osp.join(mode_folder,'layer_{}_block_{}'.format(layer,block)+'_colness.txt')
    
        stats_dict = {}
        for num_styles in [0,4,8,16]:
            stats_dict[num_styles] = []


        for idx,img_dir in enumerate(img_dirs):
            for num_styles in [0,4,8,16]:
            
                print('Processing image {}/{}, style {}'.format(idx,len(img_dirs)-1, num_styles))
            
                grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
                grads_path = osp.join(grads_folder,'grads_img_{}_layer_{}_block_{}_style_{}.txt'.format(idx,layer,block,num_styles))
            
            
                with open(grads_path,'rb') as f:
                    guided_grads = pickle.load(f)

                guided_grads = np.clip(guided_grads,0,guided_grads.max())

                grads_normld = np.uint8(255*(guided_grads - guided_grads.min())/(guided_grads.max()-guided_grads.min())
               )
                #colness = image_colorfulness(grads_normld)
                hsv = cv2.cvtColor(grads_normld, cv2.COLOR_BGR2HSV)
                hue_std = np.std(hsv[:,:,0])
                stats_dict[num_styles].append(hue_std)

                with open(stats_path, 'w') as f:
                    print(stats_dict, file=f)
        