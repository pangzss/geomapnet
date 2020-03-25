import sys
#sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from filter_extractor_random import generate_strong_filters
from skimage.transform import resize
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from guidedbp_vanilla import pipe_line
from guidedbp_layer import pipe_line as pipeline_layer
from optm_visual import optm_visual
from torchvision import models
import configparser
import argparse
import pickle
import cv2
#config 
parser = argparse.ArgumentParser(description='Filter visualization for MapNet with ResNet34')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight',
                                                    'Cambridge','stylized','Dog_and_Cat'),
                    help = 'Dataset')
parser.add_argument('--styles',type=int,choices=(0,4,8,16))
parser.add_argument('--method',type=str, choices=('guidedbp','vanilla'), help='methods for visualization')
parser.add_argument('--mode',type=str, choices=('maxima_full','maxima_patches','same_images',
                                        'image_directories_dict','one_image_for_all_blocks','optm_visual','plot_activation','patch_grid')
                                    , help='To show the full image or patches containing the maxima')
parser.add_argument('--sample_test',type=int
                                    , help='which sample image to choose when in mode one_image_for_all_blocks ')
args = parser.parse_args()


# get model
weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[args.styles])
model = get_model(weights_dir)
#model._modules['layer4'][0]._modules['conv2'].weight.data = torch.zeros_like(model._modules['layer4'][0]._modules['conv2'].weight.data)
#model = models.resnet34(pretrained=False)
model.eval()

# define root folder
root_folder = osp.join('./figs',args.dataset+'_files')
# strong filters
criterion = 'max'
filter_maxima_path = osp.join(root_folder,'filter_indices_'+criterion,'filterMaxima_'+'style'+str(args.styles)+'.pt')
img_dirs_path = osp.join(root_folder, 'filter_indices_'+criterion, 'img_dirs_'+'style'+str(args.styles)+'.txt')
if not os.path.exists(filter_maxima_path):
    # generate files of strong filters
    generate_strong_filters(model,args.dataset,filter_maxima_path,img_dirs_path)
# load strong filter files
filterMaxima = torch.load(filter_maxima_path)
with open(img_dirs_path, 'rb') as fb:
    img_dirs = pickle.load(fb)


# start guided bp/ vanilla
num_blocks = [3,4,6,3]
img_filter_indices_dict = {}
if 'maxima' in args.mode:
    for layer in range(1,4+1):
        img_filter_indices_dict['layer'+str(layer)] = {}
        for block in range(0,num_blocks[layer-1]):
            img_filter_indices_dict['layer'+str(layer)]['block'+str(block)] = {}
            # In AachenDay, there are 349 sample images. Every one of them
            # has a strongest filter in every block. Every strongest filter has its
            # own index among all the filters owned by the block.  Here the code aims to
            # pick out top k of those filters and corresponding sample images.
            
            # top k filters to visualize
            topK = 8
            # arrange filter indices in a descending order based on their activation values
            maxima_values = filterMaxima['layer'+str(layer)]['block'+str(block)][0]
            img_filter_indices = filterMaxima['layer'+str(layer)]['block'+str(block)][1]

            maxima_h2l = np.argsort(maxima_values)[::-1][:topK]
            maxima_img_idces = img_filter_indices[maxima_h2l][:,0]
            maxima_filter_idces = img_filter_indices[maxima_h2l][:,1]
            img_filter_indices_dict['layer'+str(layer)]['block'+str(block)] = img_filter_indices[maxima_h2l]
            # select the images that activate those strong filters.
            imgs_selected = []
            for idx in maxima_img_idces:
                    
                imgs_selected.append(img_dirs[int(idx)]) 
            #######################################
        
            for i,img_dir in enumerate(imgs_selected):

                dataset_path = osp.join('data', args.dataset)
                img_path = osp.join(dataset_path,img_dir)
                filter_idx = int(maxima_filter_idces[i])

                    # load an image
                img = load_image(img_path)
                # preprocess an image, return a pytorch variable
                input_img = preprocess(img)

                img = np.asarray(img.resize((224, 224)))

                mode_folder = osp.join(root_folder,args.mode)

                #guided_grads = pipe_line(input_img, model, layer, block, args.method, filter_idx = filter_idx)
                guided_grads = None
                if args.mode == 'maxima_patches' and args.method == 'guidedbp':
                    # define mode folder
                    mode_folder = osp.join(root_folder,args.mode)

                    if layer != 4 :
                        kernel_size = 50
                        if layer == 3:
                            kernel_size = 100
                        max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                        max_index = np.unravel_index(max_index_flat,(224,224))
                        patch_slice = get_patch_slice(guided_grads,max_index,kernel_size=kernel_size)
                        grads_to_save = guided_grads[:,patch_slice[0],patch_slice[1]]
                        img_to_save = img[patch_slice[0],patch_slice[1],:]
                    else:
                        grads_to_save = guided_grads
                        img_to_save = img

                elif args.mode == 'maxima_full' and args.method == 'guidedbp':
                    mode_folder = osp.join(root_folder,args.mode)
                    to_size = (112,112)
                    #[guided_grads,mask] = pipeline_layer(input_img, model, layer, block, args.method, filter_idx = filter_idx)
                    [guided_grads,mask] = pipeline_layer(input_img, model, layer, block,to_size)
                    #if layer != 4: 
                    #    kernel_size = 50
                    #    if layer == 3:
                    #        kernel_size = 100
                
                    #    max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                    #    max_index = np.unravel_index(max_index_flat,(224,224))
                    #    patch_slice = get_patch_slice(guided_grads.copy(),max_index,kernel_size=kernel_size)
                    #    grads_to_save,img_to_save = bounding_box(norm_std(guided_grads.copy()),img.copy(),patch_slice)
                    #else:
                    #    grads_to_save = norm_std(guided_grads)
                    #    img_to_save = img
                    grads_to_save = norm_std(guided_grads)
                    img_to_save = resize(img,to_size)
                    
                    #img_norml = cv2.normalize(img_to_save.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    #vis = np.uint8((mask[:, :, np.newaxis] * 0.8 + 0.2) * img_norml*255)
                    img_norml = cv2.normalize(img_to_save.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    mask = grads_to_save.sum(axis=2)
                    mask = mask / np.max(mask)
                    threshold_scale = np.percentile(mask,65)
                    mask[mask < threshold_scale] = 0.2 # binarize the mask
                    mask[mask > threshold_scale] = 1.0
                    #mask = gaussian(mask,1)
                    img_mask = np.multiply(img_norml, mask[:,:,np.newaxis])
                    img_mask = np.uint8(img_mask * 255)
                    #idx_nonzeros=np.where(np.sum(guided_grads,axis=0)!=0)
                    
                    #center = [(np.max(idx)-np.min(idx))//2+np.min(idx) for idx in idx_nonzeros]
                    #RF = [np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros] # receptive field
                    #patch_slice = get_patch_slice(guided_grads,center, RF)
                 
                    
                   
                
                file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_top'+str(i+1)
                to_folder = osp.join(mode_folder, args.method, 'style_'+str(args.styles))
                    # Save colored gradients
                if args.method == 'guidedbp':
                    save_gradient_images(grads_to_save, to_folder, file_name_to_export)
                    save_original_images(img_to_save, to_folder, file_name_to_export+'ori')
                    save_original_images(img_mask, to_folder, file_name_to_export+'mask')
                elif args.method == 'vanilla':
                    # Convert to grayscale
                    grayscale_guided_grads = convert_to_grayscale(guided_grads)
                    save_gradient_images(grayscale_guided_grads, to_folder, file_name_to_export)
                    save_original_images(img, to_folder, file_name_to_export+'ori')
                else:
                    KeyError
                # Positive and negative saliency maps
                #pos_sal, _ = get_positive_negative_saliency(guided_grads)
                

                print('Backprop completed. Layer {}, block {}, filter No.{}, top {}'.format(layer, block, filter_idx,i+1))
    with open(to_folder+'_img_filter_indices_dict.txt','wb') as fp:
        pickle.dump(img_filter_indices_dict,fp)
    
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
                print('Backprop completed. Layer {}, block {}, top {}'.format(layer, block, i+1))

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
elif args.mode == 'optm_visual':
    pass
elif args.mode == 'plot_activation':

    from activation_extractor import conv_forward
    from math import sqrt, ceil
    def vis_grid(feat_map): # feat_map: (C, H, W, 1)
        (C, H, W, B) = feat_map.shape
        cnt = int(ceil(sqrt(C)))
        G = np.ones((cnt * H + cnt, cnt * W + cnt, B), feat_map.dtype)  # additional cnt for black cutting-lines
        G *= np.min(feat_map)

        n = 0
        for row in range(cnt):
            for col in range(cnt):
                if n < C:
                    # additional cnt for black cutting-lines
                    G[row * H + row : (row + 1) * H + row, col * W + col : (col + 1) * W + col, :] = feat_map[n, :, :, :]
                    n += 1

        # normalize to [0, 1]
        G = (G - G.min()) / (G.max() - G.min())

        return G

    # visualize a layer (a feature map represented by a grid)
    def vis_layer(feat_map_grid):
        plt.clf()   # clear figure
        fig,ax = plt.subplots(ncols=1)
        ax.imshow(feat_map_grid[:, :, 0], cmap = 'gray')   # feat_map_grid: (ceil(sqrt(C)) * H, ceil(sqrt(C)) * W, 1)
        plt.axis('off')
        return fig
    
    mode_folder = osp.join(root_folder,args.mode)
    dataset_path = osp.join('data', args.dataset)

    #with open(osp.join(root_folder,'image_directories_dict/img_dirs_dict_style_0.txt'), 'rb') as fb:
     #   img_dirs_dict = pickle.load(fb)
    #img_dir = img_dirs_dict['layer3']['block2'][4]
    #img_path = osp.join(dataset_path,img_dir)
    img_path = './data/sky_patch_1.png'
    # load an image
    img = load_image(img_path)
    # preprocess an image, return a pytorch variable
    
    
    input_img = preprocess(img)
    #input_img[:,:,:,:] = 0
    #input_img[:,2,:50,:50] = 1
    img = np.asarray(img.resize((224, 224)))
    
    _,activations = conv_forward().get_conv_maps(model,input_img)
    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):
    
            actv_map = activations['layer'+str(layer)]['block'+str(block)]
            actv_map_grid = vis_grid(actv_map.data.numpy().transpose(1, 2, 3, 0))
            fig = vis_layer(actv_map_grid)

            folder = osp.join(mode_folder,'style_'+str(args.styles))
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig(osp.join(folder,'layer_'+str(layer)+'_block_'+str(block)+'.png'))
    #fig, ax= plt.subplots(ncols=1)
    #ax.imshow(activations['layer4']['block2'][0,401].data.numpy(),cmap='gray')
    #ax.set_title('Filter no.401')
    #plt.axis('off')
    #plt.show()
elif args.mode == 'patch_grid':
    mode_folder = osp.join(root_folder,args.mode)

    def patches_grid(patches,edge=None): # feat_map: (C, H, W, 1)
        # input patch : 9x3x50x50 or 9x3x100x100
        patches = patches.transpose(0,2,3,1)
        (B,H,W,C) = patches.shape
        cnt = 3
        if edge == None:
            G = np.ones((cnt * H + cnt+5, cnt * W + cnt+5 , C), patches.dtype)  # additional cnt for black cutting-lines
        else:
            edge = 5
            G = np.ones((cnt * H + cnt+edge, cnt * W + cnt+edge , C), patches.dtype)  # additional cnt for black cutting-lines
        G *= np.min(patches)

        n = 0
        for row in range(cnt):
            for col in range(cnt):
                if n < B:
                    # additional cnt for black cutting-lines
                    G[row * H + row : (row + 1) * H + row, col * W + col : (col + 1) * W + col, :] = patches[n, :, :, :]
                    n += 1

        # normalize to [0, 1]
        G = G - G.min()
        if G.max() != 0 :
            G /= G.max() 

        return G
    

    # strong filters
    criterion = 'max'
    filter_maxima_path = osp.join(root_folder,'filter_indices_'+criterion,'filterMaxima_list_'+'style'+str(args.styles)+'.pt')
    img_dirs_path = osp.join(root_folder,'filter_indices_'+criterion, 'img_dirs_'+'style'+str(args.styles)+'.txt')
    iterations = 4
    if not os.path.exists(filter_maxima_path):
        print('reach')
        # generate files of strong filters
        generate_strong_filters(model,args.dataset,filter_maxima_path,img_dirs_path,iterations=iterations,criterion=criterion)
    # load strong filter files
    filterMaxima = torch.load(filter_maxima_path)
    with open(img_dirs_path, 'rb') as fb:
        img_dirs = pickle.load(fb)

    # reference for histogram matching
    #temp = np.asarray(load_image(osp.join(mode_folder,'temp.png'))).transpose(2,0,1)
    
    
    for layer in range(4,5):
        G_grads_all = []
        G_img_all = []
        for block in range(0,num_blocks[layer-1]):
            # top k filters to visualize
            topK = 9

            G_grads = None
            img_grads = None
            for i_sample in range(iterations):

                # arrange filter indices in a descending order based on their activation values
                maxima_values = filterMaxima[i_sample]['layer'+str(layer)]['block'+str(block)][0]
                img_filter_indices = filterMaxima[i_sample]['layer'+str(layer)]['block'+str(block)][1]

                maxima_h2l = np.argsort(maxima_values)[::-1][:topK]
    
                maxima_img_idces = img_filter_indices[maxima_h2l][:,0]
                maxima_filter_idces = img_filter_indices[maxima_h2l][:,1]

                # select the images that activate those strong filters.
                imgs_selected = []
                for idx in maxima_img_idces:
                        
                    imgs_selected.append(img_dirs[int(idx)]) 
                #######################################
                if layer < 3:
                    grads_patches = np.zeros((topK,3,50,50))
                    img_patches = np.zeros((topK,3,50,50))
                elif layer == 3:
                    grads_patches = np.zeros((topK,3,100,100))
                    img_patches = np.zeros((topK,3,100,100))
                elif layer == 4:
                    grads_patches = np.zeros((topK,3,100,100))
                    img_patches = np.zeros((topK,3,100,100))
                for i,img_dir in enumerate(imgs_selected):
                   # print('top '+str(i+1),maxima_values[maxima_h2l][i])

                    dataset_path = osp.join('data', args.dataset)
                    img_path = osp.join(dataset_path,img_dir)
                    filter_idx = int(maxima_filter_idces[i])
                
                    # load an image
                    img = load_image(img_path)
                    # preprocess an image, return a pytorch variable
                    input_img = preprocess(img)

                    img = np.asarray(img.resize((224, 224)))

                    guided_grads = pipe_line(input_img, model, layer, block, args.method, filter_idx = filter_idx)
                    guided_grads = norm_std(guided_grads)
                  
                    kernel_size = 50
                    if layer != 4:
                        if layer == 3:
                            kernel_size = 100
                        
                        max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
                        max_index = np.unravel_index(max_index_flat,(224,224))
                        patch_slice = get_patch_slice(guided_grads,max_index,kernel_size=kernel_size)
                        grads_patch = guided_grads[:,patch_slice[0],patch_slice[1]].copy()
                        img_patch = img[patch_slice[0],patch_slice[1],:].transpose(2,0,1)
                    elif layer == 4:
                        grads_patch = norm_std(resize(guided_grads.copy(),(3,100,100),anti_aliasing=True))
                        img_patch = resize(img.transpose(2,0,1),(3,100,100),anti_aliasing=True)
                    #grads_patch = norm_std(grads_patch)
                
                    #grads_patch = grads_patch 
                    # - grads_patch.min()
                    #if grads_patch.max() != 0:
                    #    grads_patch /= grads_patch.max() 
                    
                    

                    #if grads_patch.shape[-1] != 50:
                
                 #       grads_patch = norm_std(resize(grads_patch,(3,50,50)))
                  #      img_patch = resize(img_patch, (3,50,50))
                    #if layer == 1 and block == 0:
                    #    grads_patch = norm_std(resize(guided_grads,(3,50,50)))
                    #    img_patch = resize(img.transpose(2,0,1),(3,50,50))

                    grads_patches[i] = grads_patch
                    img_patches[i] = img_patch
                if i_sample == 0:
                    if layer != 4:
                        G_grads = patches_grid(grads_patches)
                        G_img = patches_grid(img_patches)
                    else:
                        edge = 15
                        G_grads = patches_grid(grads_patches,edge=edge)
                        G_img = patches_grid(img_patches,edge=edge)
                else:
                    if layer != 4:
                        G_grads = np.concatenate((G_grads,patches_grid(grads_patches)),axis = 1)
                        G_img = np.concatenate((G_img,patches_grid(img_patches)),axis = 1)
                    else:
                        edge = 15
                        G_grads = np.concatenate((G_grads,patches_grid(grads_patches,edge)),axis = 1)
                        G_img = np.concatenate((G_img,patches_grid(img_patches,edge)),axis = 1)
                print('Processing Layer {}, Block {}, sample {}'.format(layer,block,i_sample))

            G_grads_all.append(G_grads.copy())
            G_img_all.append(G_img.copy())

            
            
        for idx in range(len(G_grads_all)):
            if idx == 0:
                G_grads_final = G_grads_all[idx]
                G_img_final = G_img_all[idx]
            else:
                G_grads_final = np.concatenate((G_grads_final,G_grads_all[idx]),axis=0)
                G_img_final = np.concatenate((G_img_final,G_img_all[idx]),axis=0)
        
  
        
    # normalize to [0, 1]
     #   G_grads_final = (G_grads_final - G_grads_final.min()) / (G_grads_final.max() - G_grads_final.min())
    

        path = osp.join(mode_folder,criterion, 'style_'+str(args.styles))
        if not os.path.exists(path):
                os.makedirs(path)
        G = np.concatenate((G_grads_final[:-5,:-5,:],G_img_final[:-5,:-5,:]),axis=1)
        save_original_images(G, path, 'layer'+str(layer))
        #if layer != 4:
            
          #  save_original_images(G_img_final[:-5,:-5,:], path, 'layer'+str(layer)+'_imgs')
        #else:
        #    G = np.concatenate((G_grads_final[:-edge,:-edge,:],G_img_final[:-edge,:-edge,:]),axis=1)
        #    save_original_images(G, path, 'layer'+str(layer))
            #save_original_images(G_grads_final[:-edge,:-edge,:], path, 'layer'+str(layer)+'_grads')

            #save_original_images(G_img_final[:-edge,:-edge,:], path, 'layer'+str(layer)+'_imgs')
    #fig1,ax1 = plt.subplots(ncols=1)
    #fig2,ax2 = plt.subplots(ncols=1)
    #ax1.imshow(G_grads_final) 
    #ax1.set_axis_off()
    #ax2.imshow(G_img_final)   
    #ax2.set_axis_off()
    #plt.show()


            