import sys
sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np

from utils import *
from filter_extractor import *
from guided_vanilla import pipe_line
import configparser
import argparse
import pickle

#config 
parser = argparse.ArgumentParser(description='Filter visualization for MapNet with ResNet34')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight'
                                                    'Cambridge','stylized','Dog_and_Cat'),
                    help = 'Dataset')
parser.add_argument('--styles',type=int,choices=(0,4))
parser.add_argument('--weights',type=str,help='model weights to load')
#parser.add_argument('--seed', type=int, help= 'random number generateor')
#parser.add_argument('--mode', type=str, choices=('individual','all'),help= 'visualize per block or all blocks')
#parser.add_argument('--layer', type=int, choices=(1,2,3,4), help= "choose layer to visualize when mode is 'indivisual' ")
#parser.add_argument('--block', type=int, help= "choose block to visualize when mode is 'individual'. Valid values: [0~2,0~3,0~5,0~2] (layer1 - layer4) ")
parser.add_argument('--method',type=str, choices=('guidedbp','vanilla'), help='methods for visualization')
parser.add_argument('--strong_filters',type=int, choices=(0,1), help='get strong filters or not')
args = parser.parse_args()

# if the mode is 'individual', check the block index is valid
# should have [3,4,6,3] blocks for the four layers of ResNet32.
if args.layer == 1 and args.mode == 'individual':
    assert args.block >= 0 and args.block<=2, ValueError
elif args.layer == 2 and args.mode == 'individual':
    assert args.block >= 0 and args.block<=3, ValueError
elif args.layer == 3 and args.mode == 'individual':
    assert args.block >= 0 and args.block<=5, ValueError
elif args.layer == 4 and args.mode == 'individual':
    assert args.block >= 0 and args.block<=2, ValueError

# get model
weights_dir = osp.join('../scripts/logs/stylized_models',args.weights)
model = get_model(weights_dir)

# load dataset
if args.strong_filters == 1:
    root_folder = osp.join('./figs',args.dataset+'_files')
    filter_maxima_path = osp.join(root_folder,'filterMaxima_'+'style'+str(args.styles)+'.pt')
    img_dirs_path = osp.join(root_folder, 'img_dirs_'+'style'+str(args.styles)+'.txt')
    if not os.path.exists(filter_maxima_path):
        generate_strong_filters(model,dataset,filter_maxima_path,img_dirs_path)

    # start guided bp/ vanilla
    filterMaxima = torch.load(filter_maxima_path)

    num_blocks = [3,4,6,3]
 
    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):
            # In AachenDay, there are 349 sample images. Every one of them
            # has a strongest filter in every block. Every strongest filter has its
            # own index among all the filters owned by the block.  Here the code aims to
            # pick out top k of those filters and corresponding sample images.
            topK = 8

            maxima_values = filterMaxima['layer'+str(layer)]['block'+str(block)][0]
            maxima_filter_idces = filterMaxima['layer'+str(layer)]['block'+str(block)][1]

            maxima_img_h2l = np.argsort(maxima_values)[::-1][:topK]
            maxima_img_filter_idces = maxima_filter_idces[maxima_img_h2l]

            with open(img_dirs_path, 'rb') as fb:
                img_dirs = pickle.load(fb)

            imgs_selected = []
            for idx in maxima_img_h2l:
                imgs_selected.append(img_dirs[idx]) 
            
        
            for i,img_dir in enumerate(imgs_selected):

                dataset_path = osp.join('data', args.dataset)
                img_path = osp.join(dataset_path,img_dir)
                filter_idx = maxima_img_filter_idces[i]

                pipe_line('guidedbp', model, img_path, layer, block, root_folder, args.styles
                top_idx = i,filter_idx = filter_idx)