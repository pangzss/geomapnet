import sys
sys.path.append('../')
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import numpy as np

from conv_resnet import resnet34
from deconv_resnet import Deconv_ResNet
from resize_resnet import Resize_ResNet
from feature_extraction import conv_forward
from utils import *

import cv2
from skimage import exposure
from common.train import load_state_dict

from models.posenet import PoseNet, MapNet

import configparser
import argparse
import time
#start = time.time()


def preprocess(img):
    img = np.asarray(img.resize((512, 512))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img_resized = img.copy()

    img_mean = np.mean(img)
    img_std  = np.std(img)
    img = (img - img_mean)/img_std
    img = img.transpose(2, 0, 1)    # reshape to (C, H, W)
    img = img[np.newaxis, :, :, :]  # add one dim to (1, C, H, W)
    
    return img_resized, Variable(torch.FloatTensor(img.astype(float)))
 

def load_image(filename):
    img = Image.open(filename)
    return img  

def enhance(img):
    img_copy = img.copy()
    median = np.zeros(3)
    for i in range(3):
        median[i] = np.median(img[:,:,i])
    img_copy[img_copy <= 1.2*median] = 0
    return img_copy

def visualize(layer,block,activations,deconv_model,num_maps = 1):
    assert layer>=1 and layer<=4 , "layer index must range from 1 to 4"
    actv_maps_sele = []
    deconv_imgs = []
    num_blocks_conv = [3,4,6,3]
    num_blocks_curr_layer = num_blocks_conv[layer-1]
    #for block in range(num_blocks_curr_layer):
    actv_maps = activations['layer'+str(layer)]['block'+str(block)]

    np.random.seed(layer+block)
    indices_to_sele = np.random.randint(actv_maps.shape[1], size=num_maps)

    for index in indices_to_sele:
        actv_maps_copy = actv_maps.clone()

        if index == 0:
            actv_maps_copy[:,1:,:,:] = 0
        else:
            actv_maps_copy[:,:index,:,:] = 0
            if index != actv_maps_copy.shape[1] - 1:
                actv_maps_copy[:,index + 1:, :,:] = 0
        
        deconv_output =  deconv_model(actv_maps_copy,layer,block)

        img = tn_deconv_img(deconv_output)
        
        actv_maps_sele.append(actv_maps_copy.data.numpy()[0][index])
        deconv_imgs.append(img)

    return actv_maps_sele, deconv_imgs            


#config 
parser = argparse.ArgumentParser(description='Filter visualization for MapNet')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight'
                                                    'Cambridge','stylized'),
                    help = 'Dataset')
parser.add_argument('--model',type=str,choices=('stylized','ordinary'))
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--seed', type=int, help= 'random number generateor')
parser.add_argument('--layer', type=int, help= 'choose layer to visualize')
parser.add_argument('--block', type=int, help= 'choose block to visualize')
parser.add_argument('--automode',type=int, help='automatcally save figs')
args = parser.parse_args()


settings = configparser.ConfigParser()
with open('../scripts/configs/style.ini','r') as f:
    settings.read_file(f)
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
# adapted resnet34 with forward hooks
feature_extractor = resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)

mapnet_model = MapNet(mapnet=posenet)
# load weights
loc_func = lambda storage, loc: storage
#weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
weights_dir = args.weights
checkpoint= torch.load(weights_dir,map_location=loc_func)
load_state_dict(mapnet_model,checkpoint['model_state_dict'])


####### visualization #######
#img_file = './aachen1.jpg'
path = osp.join('data', args.dataset)
img_dirs = os.listdir(path) 
seed = args.seed
np.random.seed(seed)

num_imgs = 9
n_plts = int(np.sqrt(num_imgs))

img_indices = np.random.randint(len(img_dirs),size = num_imgs)
imgs_resized = [] # containing loaded images
imgs_torch = [] # containing torch variables of images

for index in img_indices:
    img_name = osp.join(path,img_dirs[index])
    img = load_image(img_name)
    img_resized, img_torch = preprocess(img)
    imgs_resized.append(img_resized)
    imgs_torch.append(img_torch)


# load an image
#img = load_image(img_file)
#plt.figure(figsize = (10, 5))
#imgplot = plt.imshow(img)
#plt.show(block=False)
#plt.show()
# preprocess an image, return a pytorch variable
conv_resnet = mapnet_model._modules['mapnet']._modules['feature_extractor']
conv_resnet.eval()
# collect activations
fwd = conv_forward()

deconv_model_list = []
activations_list = []

#input_img = preprocess(img)
for i,input_img in enumerate(imgs_torch):
    # define conv model
    out,activations,identity_maps = fwd.get_conv_maps(conv_resnet,input_img)
    activations_list.append(activations.copy())
    
    # define deconv model
    deconv_model_list.append(Resize_ResNet(identity_maps))
    deconv_model_list[i].eval()
    # initialize deconv weights
    conv_layers = conv_resnet._modules
    deconv_layers = deconv_model_list[i]._modules

    num_blocks_conv = [3,4,6,3]
    # initialize weights
    for layer in range(1,4+1):
        conv_layer = conv_layers['layer'+str(layer)]
        deconv_layer = deconv_layers['layer'+str(4-layer+1)]
        for blk in range(0,num_blocks_conv[layer-1]):
            conv_blk = conv_layer[blk]
            deconv_blk = deconv_layer[num_blocks_conv[layer-1]-blk-1]
            
            deconv_blk._modules['deconv1'][2].weight.data = conv_blk._modules['conv2'].weight.data.flip(2,3).transpose(0,1)
            deconv_blk._modules['deconv2'][2].weight.data = conv_blk._modules['conv1'].weight.data.flip(2,3).transpose(0,1)
    deconv_layers['conv_last'][2].weight.data = conv_layers['conv1'].weight.data.flip(2,3).transpose(0,1)
    


num_blocks_conv = [3,4,6,3]

for auto_layer in range(1 + (not args.automode)*(args.layer - 1), 5 + (not args.automode)*(args.layer - 5 + 1)):
    
    
    for auto_block in range((not args.automode)*(args.block),
                            num_blocks_conv[auto_layer - 1] + (not args.automode)*(args.block - num_blocks_conv[auto_layer - 1] + 1)):
        
        print('current layer: ', auto_layer, 'current block: ', auto_block)
        actv_maps_all_imgs = []
        vis_results_all_imgs = []
        for i in range(num_imgs):
            
            activations = activations_list[i]
            deconv_resnet = deconv_model_list[i]
        
            # visualize
            # layer : 1 - 4
            # block : 3 4 6 3
            

            actv_sele,visual_results = visualize(auto_layer,auto_block,activations,deconv_resnet,num_maps = 1)
            actv_maps_all_imgs.append(*actv_sele)
            vis_results_all_imgs.append(*visual_results)
            #end = time.time()
            #print('With resize, time used is : ', end-start,'s')

            # show raw images
        fig0, ax0= plt.subplots(n_plts, n_plts)
        plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,
        right=0.9, hspace=0.05, wspace=0.0)
        for row in range(n_plts):
            for col in range(n_plts):
                ax0[row,col].imshow(imgs_resized[row*3+col])
                ax0[row,col].axis('off')
        fig0.suptitle('From data set {}'.format(args.dataset))
        fig0.savefig('figs/with_{}_model/seed{}/seed{}_raw.png'.format(args.model,seed,seed))
        # show activations and visualizations

        fig1, ax1= plt.subplots(n_plts, n_plts)
        plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,
        right=0.9, hspace=0.05, wspace=0.0)
        fig2, ax2= plt.subplots(n_plts, n_plts)
        plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,
        right=0.9, hspace=0.05, wspace=0.0)
        for row in range(n_plts):
            for col in range(n_plts):
                ax1[row,col].imshow(actv_maps_all_imgs[row*3+col],cmap='gray')
                ax1[row,col].axis('off')
                ax2[row,col].imshow(enhance(vis_results_all_imgs[row*3+col]))
                ax2[row,col].axis('off')
        fig1.suptitle('activations: layer{}, block{}'.format(auto_layer,auto_block))
        fig2.suptitle('visualizations: layer{}, block{}'.format(auto_layer,auto_block))

        if args.automode == False:
            plt.show(block=False)
            plt.show()
        
        fig1.savefig('figs/with_{}_model/seed{}/layer{}_block{}_seed{}_activations.png'.format(args.model,seed,auto_layer,auto_block,seed))
        fig2.savefig('figs/with_{}_model/seed{}/layer{}_block{}_seed{}_visualization.png'.format(args.model,seed,auto_layer,auto_block,seed))
        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)
#plt.close(fig0,fig1,fig2)