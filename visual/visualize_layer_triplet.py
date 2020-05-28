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
#from guidedbp_vanilla import pipe_line
from guidedbp_layer import GuidedBackprop
from smooth_grads import SmoothGradients
from optm_visual import optm_visual
from torchvision import models
import configparser
import argparse
import pickle
import cv2
from PIL import ImageDraw
from PIL import ImageFont
import torch.nn as nn
import torchvision.transforms as transforms
from dataset_loaders.aachen_day_night import AachenDayNight
from dataset_loaders.cambridge_triplet_visual import CambridgeTriplet
from torch.utils import data

from AdaIN import net
from AdaIN.function import adaptive_instance_normalization

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar','AachenDayNight','Cambridge'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default = ' ', help='Scene name')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load('../AdaIN/models/decoder.pth'))
vgg.load_state_dict(torch.load('../AdaIN/models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)

decoder.to(device)


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

train = False
if args.dataset == 'AachenDayNight': 
    dset = AachenDayNight(data_path, train,transform=transform)
elif args.dataset == 'Cambridge':
    dset = CambridgeTriplet(data_path, train, scene=args.scene,transform=transform, target_transform=target_transform,min_perceptual=True,style_dir='../data/pbn_test_embedding_dist.txt')
print('Loaded {:s} training data, length = {:d}'.format(args.scene if not ' ' else args.dataset, len(dset)))

batch_size=1
assert batch_size == 1, 'batch size must be one'
data_loader = data.DataLoader(dset, batch_size=batch_size, shuffle=True)
  
# get model
weights_names = ['OldHospital_100_seed0.pth.tar',
                'OldHospital_50_0.5_seed0.pth.tar',
                'OldHospital_triplet_tuple_NAN_0.5_seed0.pth.tar']
SG_list = []
param_n = 25
param_sigma_multiplier = 3
to_size = (256,256)

for name in weights_names:
    weights_dir = osp.join('../scripts/logs/stylized_models',name)
    SG_list.append(SmoothGradients(get_model(weights_dir).cuda().eval(),4,2,param_n, param_sigma_multiplier, mode = 'GBP',to_size=to_size))

# define root folder
root_folder = osp.join('./figs',args.dataset,args.scene)
mode_folder = osp.join(root_folder,'Visualize_Layer_Triplet','train' if train else 'val')

# start guided bp/ vanilla
margin = 3
num_blocks = [3,4,6,3]

layer = 4
block = 2
    
#topk = 8
#G_row = []
#discrep_row = []
#carved_row = []

for k,(data,_) in enumerate(data_loader):
    if k >= 50:
        break
    print('processing sample {}'.format(k))
    data_shape = data[0].shape
    to_shape = (-1,data_shape[-3],data_shape[-2],data_shape[-1])
    real = data[0].reshape(to_shape)

    style_stats = data[1].reshape(-1,2,512)
    style_indc = data[2].view(-1)
    stylized = None
    with torch.no_grad():
        alpha = 0.5
        content_f = vgg(real[style_indc == 1].cuda())
        style_f_stats = style_stats[style_indc == 1].unsqueeze(-1).unsqueeze(-1).cuda()
    
        feat = adaptive_instance_normalization(content_f, style_f_stats,style_stats=True)
        feat = feat *alpha + content_f * (1 -alpha)
        stylized = decoder(feat).cpu()[...,:real.shape[-2],:real.shape[-1]]
        # the output from the decoder gets padded, so only keep the portion that has
        # the same size as the original
                    

    real = real.reshape(data_shape)
    stylized = stylized[:,None,...]
    real = torch.cat([real,stylized],dim=1)
    real = real.to(device)

    G_row = []
    for j,img in enumerate(real[0]):
        img = img[None,...]
        ori_img =  inv_normalize(img[0]).cpu().numpy().transpose(1,2,0)

        G_col = []
    # grads_list = []
        for i in range(len(SG_list)):
            interrupt = 0
            grads_folder = osp.join(mode_folder,'grads_layer_{}'.format(layer))
            if not os.path.exists(grads_folder):
                os.makedirs(grads_folder)
            grads_path = osp.join(grads_folder,'grads_layer_{}_block_{}_model_{}_data_{}_img_{}.txt'.format(layer,block,i,k,j))
            if not os.path.exists(grads_path):

                guided_grads = SG_list[i].pipe_line(img)
                with open(grads_path,'wb') as f:
                    pickle.dump(guided_grads,f)
            else:
                interrupt = 1
                break
                #with open(grads_path,'rb') as f:
                #    guided_grads = pickle.load(f)
            # with open(grads_path,'rb') as f:
            #     guided_grads = pickle.load(f)
            
            ori_img = np.uint8((ori_img - ori_img.min())/(ori_img.max()-ori_img.min())*255)
            if i == 0:
                #discrep_col.append(img)
                #discrep_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
                G_col.append(ori_img)
                G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))

            guided_grads = np.clip(guided_grads,0,guided_grads.max())
            #grads_output = np.uint8(255*(guided_grads - guided_grads.min())/(guided_grads.max()-guided_grads.min()))
            grads_output = norm_std(guided_grads.copy())

            G_col.append(grads_output)
            G_col.append(np.uint8(np.ones((to_size[0],margin,3))*255))
        

        if interrupt == 1:
            break
        G_col = np.concatenate(G_col, axis=1)
        G_row.append(G_col)
        G_row.append(np.uint8(np.ones((margin,G_col.shape[1],3))*255))
    if interrupt == 1:
        continue 
    G_row = np.concatenate(G_row,axis=0)
    file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_data_'+str(k)
    to_folder = osp.join(mode_folder)
    save_original_images(G_row[:-margin,:-margin], to_folder, file_name_to_export)





