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
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight',
                                                    'Cambridge','stylized','Dog_and_Cat'),
                    help = 'Dataset')
parser.add_argument('--styles',type=int,choices=(0,4,8,16))
#parser.add_argument('--weights',type=str,help='model weights to load')
#parser.add_argument('--seed', type=int, help= 'random number generateor')
#parser.add_argument('--mode', type=str, choices=('individual','all'),help= 'visualize per block or all blocks')
#parser.add_argument('--layer', type=int, choices=(1,2,3,4), help= "choose layer to visualize when mode is 'indivisual' ")
#parser.add_argument('--block', type=int, help= "choose block to visualize when mode is 'individual'. Valid values: [0~2,0~3,0~5,0~2] (layer1 - layer4) ")
parser.add_argument('--method',type=str, choices=('guidedbp','vanilla'), help='methods for visualization')
parser.add_argument('--mode',type=str, choices=('maxima_full','maxima_patches','same_images','image_directories_dict','one_image_for_all_blocks')
                                    , help='To show the full image or patches containing the maxima')
parser.add_argument('--sample_test',type=int
                                    , help='which sample image to choose when in mode one_image_for_all_blocks ')
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
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[args.styles])
model = get_model(weights_dir)
model.eval()

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
if 'maxima' in args.mode:
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

                    if layer != 4 :
                        max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                        max_index = np.unravel_index(max_index_flat,(224,224))
                        patch_slice = get_patch_slice(guided_grads,max_index)
                        grads_to_save = guided_grads[:,patch_slice[0],patch_slice[1]]
                        img_to_save = img[patch_slice[0],patch_slice[1],:]
                    else:
                        grads_to_save = guided_grads
                        img_to_save = img

                elif args.mode == 'maxima_full' and args.method == 'guidedbp':
                    mode_folder = osp.join(root_folder,args.mode)

                    max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                    max_index = np.unravel_index(max_index_flat,(224,224))
                    patch_slice = get_patch_slice(guided_grads,max_index)
                    
                    #idx_nonzeros=np.where(np.sum(guided_grads,axis=0)!=0)
                    
                    #center = [(np.max(idx)-np.min(idx))//2+np.min(idx) for idx in idx_nonzeros]
                    #RF = [np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros] # receptive field
                    #patch_slice = get_patch_slice(guided_grads,center, RF)
                    if layer != 4:
                        grads_to_save,img_to_save = bounding_box(guided_grads.copy(),img.copy(),patch_slice)
                    else:
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
elif args.mode == 'same_images':# and args.styles != 0:
    # this mode uses the top k images obtained from the model with no style for 
    # models with nonzero styles. For the sake of comparison based a same set of images.
    mode_folder = osp.join(root_folder,args.mode)
    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):
            
            # load style 0 maximal filters
            filter_maxima_path = osp.join(root_folder,'filterMaxima_'+'style'+str(0)+'.pt')
            filterMaxima = torch.load(filter_maxima_path)

            # load style 0 maximal images
            img_dirs_path = osp.join(root_folder, 'img_dirs_'+'style'+str(0)+'.txt')
            with open(img_dirs_path, 'rb') as fb:
                img_dirs = pickle.load(fb)


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
                # set filter index as none to let the function itself find the maximal filter.
                guided_grads = pipe_line(input_img, model, layer, block, args.method, filter_idx = None)

                mode_folder = osp.join(root_folder,args.mode)

                max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                max_index = np.unravel_index(max_index_flat,(224,224))
                patch_slice = get_patch_slice(guided_grads,max_index)

                if layer != 4:
                    grads_to_save,img_to_save = bounding_box(guided_grads.copy(),img.copy(),patch_slice)
                else:
                    grads_to_save = guided_grads
                    img_to_save = img
                file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_top'+str(i+1)
                to_folder = osp.join(mode_folder, args.method, 'style_'+str(args.styles))
        
                save_gradient_images(grads_to_save, to_folder, file_name_to_export)
                save_original_images(img_to_save, to_folder, file_name_to_export+'ori')
                print('Guided backprop completed. Layer {}, block {}, top {}'.format(layer, block, i+1))

elif args.mode == 'image_directories_dict':
    # this mode uses the top k images obtained from the model with no style for 
    # models with nonzero styles. For the sake of comparison based a same set of images.
    mode_folder = osp.join(root_folder,args.mode)
    img_dirs_dict = {}
    for layer in range(1,4+1):
        img_dirs_dict['layer'+str(layer)] = {}
        for block in range(0,num_blocks[layer-1]):
            img_dirs_dict['layer'+str(layer)]['block'+str(block)] = {}
            # load style 0 maximal filters
            filter_maxima_path = osp.join(root_folder,'filterMaxima_'+'style'+str(args.styles)+'.pt')
            filterMaxima = torch.load(filter_maxima_path)

            # load style 0 maximal images
            img_dirs_path = osp.join(root_folder, 'img_dirs_'+'style'+str(args.styles)+'.txt')
            with open(img_dirs_path, 'rb') as fb:
                img_dirs = pickle.load(fb)


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
            img_dirs_dict['layer'+str(layer)]['block'+str(block)] = imgs_selected

            dict_path = osp.join(mode_folder,'img_dirs_dict_'+'style_'+str(args.styles)+'.txt')
            with open(dict_path, 'wb') as fp:
                pickle.dump(img_dirs_dict, fp)
elif args.mode == 'one_image_for_all_blocks':
    mode_folder = osp.join(root_folder,args.mode)
    dataset_path = osp.join('data', args.dataset)
    with open(osp.join(root_folder,'image_directories_dict/img_dirs_dict_style_0.txt'), 'rb') as fb:
        img_dirs_dict = pickle.load(fb)
    if args.dataset == 'AachenDay':
        sample_images = {1:img_dirs_dict['layer2']['block2'][3],
                        2:img_dirs_dict['layer2']['block0'][-1],
                        3:img_dirs_dict['layer1']['block1'][4],
                        4:'2010-10-30_17-47-25_73.jpg',
                        5:'2010-10-30_17-48-20_774.jpg',
                        6:img_dirs_dict['layer2']['block3'][-1],
                        7:img_dirs_dict['layer2']['block1'][1],
                        8:'2011-09-18_19-49-27_741.jpg',
                        }
                        #8:img_dirs_dict['layer2']['block1'][2],
                        #9:img_dirs_dict['layer3']['block1'][0]}
    elif args.dataset == 'AachenNight':
        sample_images = {1:img_dirs_dict['layer2']['block3'][0],
                         2:img_dirs_dict['layer2']['block0'][3]
                            }
   
    img_dir = sample_images[args.sample_test]
    img_path = osp.join(dataset_path,img_dir)
    # load an image
    img = load_image(img_path)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img)
    img = np.asarray(img.resize((224, 224)))
    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):

            # set filter index as none to let the function itself find the maximal filter.
            guided_grads= pipe_line(input_img, model, layer, block, args.method, filter_idx = None)

            max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
            max_index = np.unravel_index(max_index_flat,(224,224))
            patch_slice = get_patch_slice(guided_grads,max_index)

            grads_to_save,img_to_save = bounding_box(guided_grads.copy(),img.copy(),patch_slice)

            file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)
            to_folder = osp.join(mode_folder,'sample_test_'+str(args.sample_test), 'style_'+str(args.styles))
    
            save_gradient_images(grads_to_save, to_folder, file_name_to_export)
            save_original_images(img_to_save, to_folder, file_name_to_export+'ori')
            print('Guided backprop completed. Sample test {}, Layer {}, block {}'.format(args.sample_test,layer, block))