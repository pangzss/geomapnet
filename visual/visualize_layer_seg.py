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
#config 
parser = argparse.ArgumentParser(description='Filter visualization for MapNet with ResNet34')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight',
                                                    'Cambridge','stylized','Dog_and_Cat'),
                    help = 'Dataset')
parser.add_argument('--styles',type=int,choices=(0,4,8,16))

args = parser.parse_args()


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
root_folder = osp.join('./figs',args.dataset+'_files')
mode_folder = osp.join(root_folder,'visualize_layer_seg')
topk = 8
if not os.path.exists(mode_folder):
    os.makedirs(mode_folder)
if not os.path.exists(osp.join(mode_folder,'top_img_dirs.txt')):
    model_0 = get_model(osp.join('../scripts/logs/stylized_models',weights_name[0]))
    top_img_dirs = top_images(model_0,args.dataset,topk=topk)
    with open(osp.join(mode_folder,'top_img_dirs.txt'),'wb') as f:
        pickle.dump(top_img_dirs,f)
else:
    with open(osp.join(mode_folder,'top_img_dirs.txt'),'rb') as f:
        top_img_dirs = pickle.load(f)

# start guided bp/ vanilla
to_size = (112,112)
margin = 3
num_blocks = [3,4,6,3]


        

num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        img_dirs = top_img_dirs['layer'+str(layer)]['block'+str(block)]

        G_row = []
        discrep_row = []
        carved_row = []
        hist_row = []

        
        for top_i,img_dir in enumerate(img_dirs):
            G_col = []
            discrep_col = [] 
            carved_col = []
            hist_col = []
            grads_list = []
            grads_normld_list = []
            label_list = []
            for num_styles in [0,4,8,16]:
            
                print("Working with the top {} image. Current model trained with {} styles".format(top_i+1,num_styles))


                dataset_path = osp.join('data', args.dataset)
                img_path = osp.join(dataset_path,img_dir)
                mask_path = osp.join(root_folder,'binary_masks',img_dir.split('.')[0]+'_mask.png')
                seg_path = osp.join(root_folder,'semantic_masks',img_dir.split('.')[0]+'.txt')
                '''
                lists_folder = osp.join(mode_folder,'grads_lists_layer_{}'.format(layer))
                if not os.path.exists(lists_folder):
                    os.makedirs(lists_folder)
                grads_list_path = osp.join(lists_folder,'grads_list_top_{}_layer_{}_block_{}_style_{}.txt'.format(top_i+1,layer,block,num_styles))
                '''
                grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
                if not os.path.exists(grads_folder):
                    os.makedirs(grads_folder)
                grads_path = osp.join(grads_folder,'grads_top_{}_layer_{}_block_{}_style_{}.txt'.format(top_i+1,layer,block,num_styles))
                # load an image
                img = load_image(img_path)
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

                label_list.append(label)
                # preprocess an image, return a pytorch variable
                input_img = preprocess(img)
                
                img = np.asarray(img.resize(to_size))
                if num_styles == 0:
                    #discrep_col.append(img)
                    #discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                    G_col.append(img)
                    G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                    G_col.append(img_label)
                    G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                
                if not os.path.exists(grads_path):
                    guided_grads = pipeline_layer(input_img, models[num_styles], layer, block,to_size)
                    with open(grads_path,'wb') as f:
                        pickle.dump(guided_grads,f)
                else:
                    with open(grads_path,'rb') as f:
                        guided_grads = pickle.load(f)
                guided_grads = np.clip(guided_grads,0,guided_grads.max())
                grads_list.append(guided_grads)
                grads_normld_list.append((guided_grads - guided_grads.min())/(guided_grads.max()-guided_grads.min()))


            factor = 1#np.sum(np.stack(grads_normld_list))
            for i,grads in enumerate(grads_list):
                label = label_list[i]
                grads_output = norm_std(grads.copy())
                grads_normld = (grads - grads.min())/(grads.max()-grads.min())

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


                grads_output = Image.fromarray(grads_output)
                draw = ImageDraw.Draw(grads_output)
                font = ImageFont.truetype("./open-sans/OpenSans-Regular.ttf", 8)
            
                draw.text((0, 0),"Building={:.3f}\nLight={:.3f}\nNear sky={:.3f}\nOthers={:.3f}"\
                    .format(building_grad,light_grad,near_sky_grad,others_grad),(255,255,255),font = font)

                grads_output = np.asarray(grads_output)
                G_col.append(grads_output)
                if num_styles == 0:
                    G_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))
                if num_styles == 4 or num_styles == 8:
                    G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                #if num_styles == 16:
                #    G_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))



            
            ref = grads_list[0]
            boost_list = []
            loss_list = []
            for i in range(1,4):

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



            normld = norm_std(np.concatenate(discrep_col,axis=1),scale=0.1)
            discrep_col = []
            for i in range(3):
                normld_item = normld[:,to_size[1]*i:to_size[1]*(i+1),:]
                
                normld_item = Image.fromarray(normld_item)
                draw = ImageDraw.Draw(normld_item)
                #font = ImageFont.truetype("sans-serif.ttf", 16)
                font = ImageFont.truetype("./open-sans/OpenSans-Regular.ttf", 8)
            
                draw.text((0, 0),"Boost (red)={:.2f}\nLoss (blue)={:.2f}\nTotal={:.2f}".format(boost_list[i],loss_list[i],boost_list[i]+loss_list[i]),(255,255,255),font = font)
                normld_item = np.array(normld_item)

                discrep_col.append(normld_item)
                if i != 2:
                    discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                else:
                    discrep_col.append(np.uint8(np.ones((to_size[0],2*margin,3))*255))



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
            discrep_col = np.concatenate(discrep_col,axis=1)
            carved_col = np.concatenate(carved_col,axis=1)
            #hist_col = np.concatenate(hist_col,axis=1)
            #save_original_images(hist_col, osp.join(mode_folder), 'layer_'+str(layer)+'_block_'+str(block)+'_top_'+str(top_i))

           # G_col = np.concatenate((G_col,discrep_col),axis=1)
          #  G_col = Image.fromarray(G_col).convert('RGBA')
           # G_col = np.array(G_col).astype(np.uint8)
            #G_col = np.concatenate((G_col,carved_col),axis=1)

            G_row.append(G_col)
            #hist_row.append(hist_row)
            if top_i is not topk-1:
                G_row.append(np.uint8(np.ones((margin,G_col.shape[1],3))*255))
                #hist_row.append(np.uint8(np.ones((margin,hist_col.shape[1],3))*255))

            #carved_col = np.concatenate(carved_col,axis=1)
            #carved_row.append(carved_col)
            #if top_i is not topk-1:
            #    carved_row.append(np.uint8(np.ones((margin,G_col.shape[1],4))*255))
        #discrep_row = np.concatenate(discrep_row,axis=0)
        G_row = np.concatenate(G_row,axis=0)
        #hist_row = np.concatenate(hist_row,axis=0)
        #carved_row = np.uint8(np.concatenate(carved_row,axis=0))

        file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)
        to_folder = osp.join(mode_folder)

        save_original_images(G_row, to_folder, file_name_to_export)
        #save_original_images(hist_row, to_folder, file_name_to_export+'_hist')



