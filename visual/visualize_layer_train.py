import sys
#sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from top_imgs import top_images
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
from PIL import ImageDraw
from PIL import ImageFont

import torchvision.transforms as transforms
from dataset_loaders.aachen_day_night import AachenDayNight
from torch.utils import data

transform = transforms.Compose([
transforms.Resize(224),
transforms.RandomCrop((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)
data_path = '../data/deepslam_data/AachenDayNight'
train = True
dset = AachenDayNight(data_path, train,transform=transform)
print('Loaded AachenDayNight training data, length = {:d}'.format(
len(dset)))
data_loader = data.DataLoader(dset, batch_size=8, shuffle=True)
batch_count = 0
N = 1

inputs = []
for batch in data_loader:

    ori_imgs = [inv_normalize(batch[0][i].clone()).numpy().transpose(1,2,0) for i in range(batch[0].shape[0])]
    #plt.imshow(np.transpose(ori_img[0], (1,2,0)), interpolation='nearest')
    #plt.show()
    inputs = batch[0]
    batch_count += 1
    if batch_count >= N:
        break


# get model
weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       

# define root folder
root_folder = osp.join('./figs','Aachen_train'+'_files')
mode_folder = osp.join(root_folder,'visualize_layer')


# start guided bp/ vanilla
to_size = (112,112)
margin = 3
num_blocks = [3,4,6,3]

layer = 4
block = 2
    
topk = 8
G_row = []
discrep_row = []
carved_row = []
for top_i,input_img in enumerate(inputs):
    input_img = input_img[None,:,:,:]

    G_col = []
    discrep_col = [] 
    carved_col = []
    grads_list = []
    for num_styles in [0,4,8,16]:
        print("Working with the top {} image. Current model trained with {} styles".format(top_i+1,num_styles))
        weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[num_styles])
        model = get_model(weights_dir)
        model.eval()

    
        grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
        if not os.path.exists(grads_folder):
            os.makedirs(grads_folder)
        grads_path = osp.join(grads_folder,'grads_top_{}_layer_{}_block_{}_style_{}.txt'.format(top_i+1,layer,block,num_styles))
        
        img = ori_imgs[top_i]
        img = resize(img,to_size,anti_aliasing=True)
        img = np.uint8((img - img.min())/(img.max()-img.min())*255)
        if num_styles == 0:
            #discrep_col.append(img)
            #discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
            G_col.append(img)
            G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
        '''
        if not os.path.exists(grads_list_path):

            guided_grads_list = pipeline_layer(input_img, model, layer, block,to_size)
            with open(grads_list_path,'wb') as f:
                pickle.dump(guided_grads_list,f)
        else:
            with open(grads_list_path,'rb') as f:
                guided_grads_list = pickle.load(f)
        


        guided_grads = 0
        for i in range(len(guided_grads_list)):
            guided_grads += guided_grads_list[i]
        '''
        
        if not os.path.exists(grads_path):

            guided_grads = pipeline_layer(input_img, model, layer, block,to_size)
            with open(grads_path,'wb') as f:
                pickle.dump(guided_grads,f)
        else:
            with open(grads_path,'rb') as f:
                guided_grads = pickle.load(f)
        guided_grads = np.clip(guided_grads,0,guided_grads.max())
        grads_output = norm_std(guided_grads.copy())
        grads_list.append((guided_grads - guided_grads.min())/(guided_grads.max()-guided_grads.min()+1e-05))

        G_col.append(grads_output)
        if num_styles == 0:
            G_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))
        if num_styles == 4 or num_styles == 8:
            G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
        if num_styles == 16:
            G_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))

    ref = grads_list[0]
    boost_list = []
    loss_list = []
    for i in range(1,4):

        #discrep = np.abs(ref-grads_list[i])
        #discrep = np.sum(discrep,axis=2)
        #discrep = norm_std(discrep*1.0)
        #discrep_col.append(discrep)
        #if i!= 3:
        #    discrep_col.append(np.uint8(np.ones((to_size[0],margin))*255))
        heat_map = np.zeros((to_size[0],to_size[1],3))
        
        discrep =  grads_list[i].sum(axis=2) - ref.sum(axis=2)
        #discrep = norm_std(discrep*1.0)
        lambda_ = 1.5
        indices_pos = np.nonzero(discrep>=0)

        boost = np.sqrt(np.sum(discrep[indices_pos[0],indices_pos[1]]**2))
        boost_list.append(boost)
        heat_map[indices_pos[0],indices_pos[1],0] += lambda_*(discrep[indices_pos[0],indices_pos[1]])**2

        indices_neg = np.nonzero(discrep<0)

        loss = np.sqrt(np.sum(discrep[indices_neg[0],indices_neg[1]]**2))
        loss_list.append(loss)
        heat_map[indices_neg[0],indices_neg[1],2] += lambda_*(discrep[indices_neg[0],indices_neg[1]])**2
        heat_map[indices_neg[0],indices_neg[1],1] += 0.5*lambda_*(discrep[indices_neg[0],indices_neg[1]])**2
        discrep_col.append(heat_map)

        # carving 
        heatmap_ns = Image.fromarray(norm_std(heat_map,scale=0.125)).convert('RGBA')
        heatmap_ns = np.array(heatmap_ns)
        heatmap_ns[:,:,3] = 0.7*255
        heatmap_ns = Image.fromarray((heatmap_ns).astype(np.uint8))
        # Apply heatmap on image
        heatmap_on_image = Image.new("RGBA", Image.fromarray(img).size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, Image.fromarray(img).convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap_ns)
        carved_col.append(heatmap_on_image)
        if i!= 3:
            carved_col.append(np.uint8(np.ones((to_size[0],margin,4))*255))

    carved_col = np.concatenate(carved_col,axis=1)

    normld = norm_std(np.concatenate(discrep_col,axis=1),scale=0.1)
    discrep_col = []
    for i in range(3):
        normld_item = normld[:,to_size[1]*i:to_size[1]*(i+1),:]
        
        normld_item = Image.fromarray(normld_item)
        draw = ImageDraw.Draw(normld_item)
        #font = ImageFont.truetype("sans-serif.ttf", 16)
        font = ImageFont.truetype("./open-sans/OpenSans-Regular.ttf", 10)
    
        draw.text((0, 0),"Boost (red)={:.2f}\nLoss (blue)={:.2f}".format(boost_list[i],loss_list[i]),(255,255,255),font = font)
        normld_item = np.array(normld_item)

        discrep_col.append(normld_item)
        if i != 2:
            discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
        else:
            discrep_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))

    discrep_col = np.concatenate(discrep_col,axis=1)

    #discrep_row.append(discrep_col)
    #if top_i is not topk-1:
    #    discrep_row.append(np.uint8(np.ones((margin,discrep_col.shape[1],3))*255))
    
   
    #normld = norm_std(np.concatenate(G_col[2:],axis=1))
    #G_col = G_col[:2]
    #for i in range(4):
    #    G_col.append(normld[:,to_size[1]*i:to_size[1]*(i+1),:])
    #    if i != 3:
    #        G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
    G_col = np.concatenate(G_col, axis=1)
    
    G_col = np.concatenate((G_col,discrep_col),axis=1)
    G_col = Image.fromarray(G_col).convert('RGBA')
    G_col = np.array(G_col).astype(np.uint8)
    G_col = np.concatenate((G_col,carved_col),axis=1)

    G_row.append(G_col)
    if top_i is not topk-1:
        G_row.append(np.uint8(np.ones((margin,G_col.shape[1],4))*255))

    #carved_col = np.concatenate(carved_col,axis=1)
    #carved_row.append(carved_col)
    #if top_i is not topk-1:
    #    carved_row.append(np.uint8(np.ones((margin,G_col.shape[1],4))*255))
#discrep_row = np.concatenate(discrep_row,axis=0)
G_row = np.concatenate(G_row,axis=0)
#carved_row = np.uint8(np.concatenate(carved_row,axis=0))

file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)
to_folder = osp.join(mode_folder)

save_original_images(G_row, to_folder, file_name_to_export)
#save_original_images(carved_row, to_folder, file_name_to_export+'_carved')



