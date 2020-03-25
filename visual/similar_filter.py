
import os.path as osp

import csv
import numpy as np
from utils import *
from skimage.transform import resize
from torchvision.utils import make_grid
from guidedbp_vanilla import pipe_line
import pickle

def get_nearest_filters(weights_base,weights_var):
    num_filters = weights_base.shape[0]

    dist_mat = np.zeros((num_filters,num_filters))
    for i in range(num_filters):
        candidate = weights_base[i]
        sqre = np.sqrt(np.square(candidate - weights_var).sum(axis=(1,2,3)))
        dist_mat[i,:] = sqre
    indices = np.argmin(dist_mat,axis=1)
    min_vals = np.zeros(num_filters)
    for i,idx in enumerate(indices):
        min_vals[i] = dist_mat[i,idx]
    
    matches = [(indices[i],min_vals[i]) for i in range(num_filters)]


    return matches

def get_matches(model_0,model_4,model_8,model_16):
    num_blocks = [3,4,6,3]
    matched_indices = {} 
    for layer in range(1,4+1):
        matched_indices['layer'+str(layer)]={}
        for block in range(0,num_blocks[layer-1]):
            matched_indices['layer'+str(layer)]['block'+str(block)] = {}
            conv_weights_0 = model_0._modules['layer'+str(layer)][block]._modules['conv2'].weight.data.numpy()
            conv_weights_4 = model_4._modules['layer'+str(layer)][block]._modules['conv2'].weight.data.numpy()
            conv_weights_8 = model_8._modules['layer'+str(layer)][block]._modules['conv2'].weight.data.numpy()
            conv_weights_16 = model_16._modules['layer'+str(layer)][block]._modules['conv2'].weight.data.numpy()

            matched_indices['layer'+str(layer)]['block'+str(block)][4] = get_nearest_filters(conv_weights_0, conv_weights_4) 
            matched_indices['layer'+str(layer)]['block'+str(block)][8] = get_nearest_filters(conv_weights_0, conv_weights_8) 
            matched_indices['layer'+str(layer)]['block'+str(block)][16] = get_nearest_filters(conv_weights_0, conv_weights_16) 

            print('process layer {}, block {}'.format(layer,block))
    return matched_indices


    
def patches_grid(patches,edge=None): # feat_map: (C, H, W, 1)
        # input patch : 9x3x50x50 or 9x3x100x100
        patches = patches.transpose(0,2,3,1)
        (B,H,W,C) = patches.shape
        cnt = B

        G = np.ones((cnt * H + cnt, W + 3 , C), patches.dtype)  # additional cnt for black cutting-lines
        
        G *= np.min(patches)

        n = 0
        for row in range(cnt):
            if n < B:
                # additional cnt for black cutting-lines
                G[row * H + row : (row + 1) * H + row, :W, :] = patches[n, :, :, :]
                n += 1

        # normalize to [0, 1]
        G = G - G.min()
        if G.max() != 0 :
            G = G / G.max() 

        return G

def visualization():

    weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                    4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                    8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                    16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       

    model_path = '../scripts/logs/stylized_models'

    models = {}
    for num_styles in [0,4,8,16]:
        models[num_styles] = get_model(osp.join(model_path,weights_name[0]))

    if not os.path.exists('matched_indices.txt'):
        matched_indices = get_matches(models[0],models[4],models[8],models[16])
        with open('matched_indices.txt', 'wb') as f:  # Just use 'w' mode in 3.x
            pickle.dump(matched_indices,f)
    else:
        with open('matched_indices.txt','rb') as f:
            matched_indices = pickle.load(f)
    data = 'AachenDay'
    path = osp.join('data', data)
    img_dirs = os.listdir(path)
    selected_img_path = img_dirs[int(np.random.permutation(len(img_dirs))[0])]
    selected_img = load_image(osp.join(path,selected_img_path))
    # preprocess an image, return a pytorch variable
    input_img = preprocess(selected_img)
    selected_img = np.asarray(selected_img.resize((224, 224)))

    layer = 3
    block = 0

    matches = {}

    for num_styles in [4,8,16]:
        matches[num_styles] = matched_indices['layer'+str(layer)]['block'+str(block)][num_styles]

    num_filters = matches[4].shape[0]
    selected_num_filters = 8
    selected_idces = np.random.permutation(num_filters)[:selected_num_filters]
   
    guided_grads_dict = {}
    img_dict = {}
    for num_styles in [0,4,8,16]:
        guided_grads_dict[num_styles] = []
        img_dict[num_styles] = []

    for idx in selected_idces:
        for num_styles in [0,4,8,16]:
            if num_styles == 0:
                guided_grads = pipe_line(input_img, models[num_styles], layer, block, 'guidedbp', filter_idx = idx)
            else:
                guided_grads = pipe_line(input_img, models[num_styles], layer, block, 'guidedbp', filter_idx = matches[num_styles][idx])
            guided_bp = norm_std(guided_grads)

            max_index_flat = np.argmax(np.sum(guided_grads,axis=0))
            max_index = np.unravel_index(max_index_flat,(224,224))

            if layer < 3:
                kernel_size = 50
            elif layer == 3:
                kernel_size = 100
            else:
                kernel_size = 224

            patch_slice = get_patch_slice(guided_grads,max_index,kernel_size = kernel_size)
            grads_patch = guided_grads[:,patch_slice[0],patch_slice[1]].copy()
            img_patch = selected_img[patch_slice[0],patch_slice[1],:].transpose(2,0,1)

            if layer == 4:
                grads_patch = norm_std(resize(guided_grads.copy(),(3,100,100),anti_aliasing=True))
                img_patch = resize(img_patch,(3,100,100),anti_aliasing=True)

            guided_grads_dict[num_styles].append(grads_patch)
            img_dict[num_styles].append(img_patch)

    G_dict = {}
    for num_styles in [0,4,8,16]:
        guided_grads_array = np.stack(guided_grads_dict[num_styles],0)
        img_array = np.stack(img_dict[num_styles],0)
        G_grads = patches_grid(guided_grads_array)
        G_img = patches_grid(img_array)
        G_dict[num_styles] = np.concatenate((G_img[:,:-3],G_grads),1)
    

    save_path = './figs/'+data+'_files/similar_filters'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    G = np.concatenate((G_dict[0],G_dict[4],G_dict[8],G_dict[16]),axis=1)
    save_original_images(G[:,:-3],save_path,'layer_'+str(layer)+'_block_'+str(block))

if __name__ == "__main__":
    weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                    4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                    8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                    16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       

    model_path = '../scripts/logs/stylized_models'

    models = {}
    for num_styles in [0,4,8,16]:
        models[num_styles] = get_model(osp.join(model_path,weights_name[0]))

    
    matches_idx_val = get_matches(models[0],models[4],models[8],models[16])
    with open('matches_idx_val', 'wb') as f:  # Just use 'w' mode in 3.x
        pickle.dump(matches_idx_val,f)
    print(matches_idx_val)