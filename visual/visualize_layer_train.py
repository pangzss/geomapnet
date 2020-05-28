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
from dataset_loaders.cambridge import Cambridge
from torch.utils import data

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar','AachenDayNight','Cambridge'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default = ' ', help='Scene name')
args = parser.parse_args()

transform = transforms.Compose([
transforms.Resize((256,256)),
#transforms.RandomCrop((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])

inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
data_path = osp.join('..','data/deepslam_data',args.dataset)

train = True
if args.dataset == 'AachenDayNight': 
    dset = AachenDayNight(data_path, train,transform=transform)
elif args.dataset == 'Cambridge':
    dset = Cambridge(data_path, train, scene=args.scene,transform=transform, target_transform=target_transform)
print('Loaded {:s} training data, length = {:d}'.format(args.scene if not ' ' else args.dataset, len(dset)))

batch_size=1
assert batch_size == 1, 'batch size must be one'
data_loader = data.DataLoader(dset, batch_size=batch_size, shuffle=False)
  
# get model
weights_names = ['ShopFacade_100_seed1.pth.tar',
                'ShopFacade_50_0.5_seed0.pth.tar',
                'ShopFacade_triplet_tuple_NAN_0.5_seed8.pth.tar']  
model_list = []
for name in weights_names:
    weights_dir = osp.join('../scripts/logs/stylized_models',name)
    model_list.append(get_model(weights_dir).cuda().eval())

# define root folder
root_folder = osp.join('./figs',args.dataset,args.scene)
mode_folder = osp.join(root_folder,'visualize_layer')

# start guided bp/ vanilla
to_size = (256,256)
margin = 3
num_blocks = [3,4,6,3]

layer = 4
block = 2
    
#topk = 8
#G_row = []
#discrep_row = []
#carved_row = []
for k,(data,_) in enumerate(data_loader):
    img = data[0]
    ori_img =  inv_normalize(img[0]).cpu().numpy().transpose(1,2,0)
    G_col = []
   # grads_list = []
    for i,model in enumerate(model_list):
        grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
        if not os.path.exists(grads_folder):
            os.makedirs(grads_folder)
        grads_path = osp.join(grads_folder,'grads_layer_{}_block_{}_model_{}_data_{}.txt'.format(layer,block,i,k))
        if not os.path.exists(grads_path):

            guided_grads = pipeline_layer(img, model, layer, block,to_size=to_size)
            with open(grads_path,'wb') as f:
                pickle.dump(guided_grads,f)
        else:
            continue
           # with open(grads_path,'rb') as f:
           #     guided_grads = pickle.load(f)
           
        ori_img = np.uint8((ori_img - ori_img.min())/(ori_img.max()-ori_img.min())*255)
        if i == 0:
            #discrep_col.append(img)
            #discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
            G_col.append(ori_img)
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

        guided_grads = np.clip(guided_grads,0,guided_grads.max())
        grads_output = norm_std(guided_grads.copy())

        G_col.append(grads_output)
        G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))

    if len(G_col) < len(model_list):
        continue

    G_col = np.concatenate(G_col, axis=1)

    file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_data_'+str(k)
    to_folder = osp.join(mode_folder)

    save_original_images(G_col[:,:-margin], to_folder, file_name_to_export)




