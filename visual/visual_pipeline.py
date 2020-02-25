import sys
#sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np

from utils import *
from filter_extractor import generate_strong_filters
from guidedbp_vanilla import pipe_line
import configparser
import argparse
import pickle

#config 
parser = argparse.ArgumentParser(description='Filter visualization for MapNet with ResNet34')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight'
                                                    'Cambridge','stylized','Dog_and_Cat'),
                    help = 'Dataset')
parser.add_argument('--styles',type=int,choices=(0,4))
#parser.add_argument('--weights',type=str,help='model weights to load')
#parser.add_argument('--seed', type=int, help= 'random number generateor')
#parser.add_argument('--mode', type=str, choices=('individual','all'),help= 'visualize per block or all blocks')
#parser.add_argument('--layer', type=int, choices=(1,2,3,4), help= "choose layer to visualize when mode is 'indivisual' ")
#parser.add_argument('--block', type=int, help= "choose block to visualize when mode is 'individual'. Valid values: [0~2,0~3,0~5,0~2] (layer1 - layer4) ")
parser.add_argument('--method',type=str, choices=('guidedbp','vanilla'), help='methods for visualization')
parser.add_argument('--mode',type=str, choices=('maxima_full','maxima_patches')
                                    , help='To show the full image or patches containing the maxima')
args = parser.parse_args()

# if the mode is 'individual', check the block index is valid
# should have [3,4,6,3] blocks for the four layers of ResNet32.
#if args.layer == 1 and args.mode == 'individual':
#    assert args.block >= 0 and args.block<=2, ValueError
#elif args.layer == 2 and args.mode == 'individual':
#    assert args.block >= 0 and args.block<=3, ValueError
#elif args.layer == 3 and args.mode == 'individual':
#    assert args.block >= 0 and args.block<=5, ValueError
#elif args.layer == 4 and args.mode == 'individual':
#    assert args.block >= 0 and args.block<=2, ValueError

# get model
weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'}       
weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[args.styles])
model = get_model(weights_dir)


# define root folder
root_folder = osp.join('./figs',args.dataset+'_files')
# strong filters
filter_maxima_path = osp.join(root_folder,'filterMaxima_'+'style'+str(args.styles)+'.pt')
img_dirs_path = osp.join(root_folder, 'img_dirs_'+'style'+str(args.styles)+'.txt')
if not os.path.exists(filter_maxima_path):
    # generate files of strong filters
    generate_strong_filters(model,args.dataset,filter_maxima_path,img_dirs_path)
# load strong filter files
filterMaxima = torch.load(filter_maxima_path)
with open(img_dirs_path, 'rb') as fb:
    img_dirs = pickle.load(fb)


# start guided bp/ vanilla
num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        # In AachenDay, there are 349 sample images. Every one of them
        # has a strongest filter in every block. Every strongest filter has its
        # own index among all the filters owned by the block.  Here the code aims to
        # pick out top k of those filters and corresponding sample images.
        
        # top k filters to visualize
        topK = 8
        # arrange filter indices in a descending order based on their activation values
        maxima_values = filterMaxima['layer'+str(layer)]['block'+str(block)][0]
        maxima_filter_idces = filterMaxima['layer'+str(layer)]['block'+str(block)][1]

        maxima_img_h2l = np.argsort(maxima_values)[::-1][:topK]
        maxima_img_filter_idces = maxima_filter_idces[maxima_img_h2l]
        # select the images that activate those strong filters.
        imgs_selected = []
        for idx in maxima_img_h2l:
            imgs_selected.append(img_dirs[idx]) 
        #######################################
    
        for i,img_dir in enumerate(imgs_selected):

            dataset_path = osp.join('data', args.dataset)
            img_path = osp.join(dataset_path,img_dir)
            filter_idx = maxima_img_filter_idces[i]

                # load an image
            img = load_image(img_path)
            # preprocess an image, return a pytorch variable
            input_img = preprocess(img)

            img = np.asarray(img.resize((224, 224)))

            guided_grads = pipe_line(input_img, model, layer, block, args.method, filter_idx = filter_idx)
            if args.mode == 'maxima_patches' and args.method == 'guidedbp':
                # define mode folder
                mode_folder = osp.join(root_folder,args.mode)
             
                max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                max_index = np.unravel_index(max_index_flat,(224,224))
                patch_slice = get_patch_slice(guided_grads,max_index)
                grads_to_save = guided_grads[:,patch_slice[0],patch_slice[1]]
                img_to_save = img[patch_slice[0],patch_slice[1],:]
            elif args.mode == 'maxima_full' and args.method == 'guidedbp':
                mode_folder = osp.join(root_folder,args.mode)

                grads_to_save = guided_grads
                img_to_save = img
            
            file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_top'+str(i+1)
            to_folder = osp.join(mode_folder, args.method, 'style_'+str(args.styles))
                # Save colored gradients
            if args.method == 'guidedbp':
                save_gradient_images(grads_to_save, to_folder, file_name_to_export)
                save_original_images(img_to_save, to_folder, file_name_to_export+'ori')
            elif args.method == 'vanilla':
                # Convert to grayscale
                grayscale_guided_grads = convert_to_grayscale(guided_grads)
                save_gradient_images(grayscale_guided_grads, to_folder, file_name_to_export)
                save_original_images(img, to_folder, file_name_to_export+'ori')
            else:
                KeyError
            # Positive and negative saliency maps
            #pos_sal, _ = get_positive_negative_saliency(guided_grads)
            

            print('Guided backprop completed. Layer {}, block {}, filter No.{}, top {}'.format(layer, block, filter_idx,i+1))