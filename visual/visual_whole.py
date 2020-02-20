import sys
sys.path.append('../')

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from conv_resnet import resnet34
from transposeConv_resnet import transposeConv_resnet
from resizeConv_resnet import resizeConv_resnet
from feature_extraction import conv_forward
from utils import *

import cv2
from skimage import exposure
from skimage.filters import median, gaussian
from skimage.morphology import disk

from common.train import load_state_dict
# import mapnet 

from models.posenet import PoseNet, MapNet
#from torchvision import models 
import configparser
import time

start = time.time()
device = torch

def preprocess(img):
    img = np.asarray(img.resize((224, 224))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)
    print(img.shape)
    img = img[None,:,:,:]
    return img

def load_image(filename):
    img = Image.open(filename)
    return img
    
def get_patch_slice(actv_map,max_index,kernel_size=3):
    H,W = actv_map.shape[2:]
    h,w = max_index[2:]

    radius = (kernel_size - 1)/2

    left = w - radius
    right = w + radius + 1
    w_slice = slice( int(left*(left>0)), int(right - (right-W)*(right>W)))

    up = h - radius
    down = h + radius + 1
    h_slice = slice(int(up*(up>0)), int(down - (down-H)*(down>H)))

    return (max_index[0],max_index[1],h_slice, w_slice)

def get_strongest_filters(activation_img, top=3):
    
    activation_img = activation_img.detach().numpy()
    # Find maximum activation for each filter for a given image
    activation_img = np.nanmax(activation_img, axis=3)
    activation_img = np.nanmax(activation_img, axis=2)

    activation_img = activation_img.sum(0)

    # Make activations 1-based indexing
    #activation_img = np.insert(activation_img, 0, 0.0)

    #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
    #  each entry corresponds to the maximum/summed activation of each filter for a given image

    top_filters = activation_img.argsort()[-top:]
    return list(top_filters)

def visualize(layer,block,activations,deconv_model,num_maps = 1):
    assert layer>=1 and layer<=4 , "layer index must range from 1 to 4"
    actv_maps_sele = []
    deconv_imgs = []
    num_blocks_conv = [3,4,6,3]
    num_blocks_curr_layer = num_blocks_conv[layer-1]
    #for block in range(num_blocks_curr_layer):
    actv_maps = activations['layer'+str(layer)]['block'+str(block)]
    #_,indices_to_sele = torch.topk(torch.norm(actv_maps[0],dim=(1,2)),k=9)
    
    indices_to_sele = get_strongest_filters(actv_maps, top=9)

    for index in indices_to_sele:
        actv_maps_copy = actv_maps.clone().detach().numpy()
        
        # set other feature maps to zero
        new_maps = np.zeros_like(actv_maps_copy)
        new_maps[0, index - 1] = actv_maps_copy[0, index - 1]

        # Set other activations in same layer to zero
        max_index_flat = np.nanargmax(new_maps)
        max_index = np.unravel_index(max_index_flat, new_maps.shape)
        
        max_map = np.zeros_like(new_maps)

        max_slice = get_patch_slice(new_maps,max_index,kernel_size=int(24/layer))
        max_map[max_slice] = new_maps[max_slice]
        max_map = torch.from_numpy(max_map)
    
        pool_indices = activations['maxpool']['pool_indices']
        #deconv_output =  deconv_model(max_map,layer,block, pool_indices)
        deconv_output =  deconv_model(torch.from_numpy(new_maps),layer,block, pool_indices)
        #img,coors = tn_deconv_img(deconv_output)
        img = tn_deconv_img(deconv_output)
        actv_maps_sele.append(max_map[0][index].numpy())
        deconv_imgs.append([img])

    return actv_maps_sele, deconv_imgs            


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
weights_dir = './logs/stylized_models/AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar'
checkpoint= torch.load(weights_dir,map_location=loc_func)
load_state_dict(mapnet_model,checkpoint['model_state_dict'])


####### visualization #######
#img_file = './aachen1.jpg'
img_file = './imgs/4.jpg'
# load an image
img = load_image(img_file)
# preprocess an image, return a pytorch variable
input_img = preprocess(img)
# define conv model
conv_resnet = mapnet_model._modules['mapnet']._modules['feature_extractor']
conv_resnet.eval()
# collect activations
fwd = conv_forward()
out,activations = fwd.get_conv_maps(conv_resnet,input_img)
# define deconv model
deconv_resnet = resizeConv_resnet()
deconv_resnet.eval()
# initialize deconv weights
conv_layers = conv_resnet._modules
deconv_layers = deconv_resnet._modules

num_blocks_conv = [3,4,6,3]
for layer in range(1,4+1):
    conv_layer = conv_layers['layer'+str(layer)]
    deconv_layer = deconv_layers['layer'+str(4-layer+1)]
    for blk in range(0,num_blocks_conv[layer-1]):
        conv_blk = conv_layer[blk]
        deconv_blk = deconv_layer[num_blocks_conv[layer-1]-blk-1]
        
        if deconv_resnet.name == 'transposeConv':
            deconv_blk._modules['deconv1'].weight.data = conv_blk._modules['conv2'].weight.data
            deconv_blk._modules['deconv1'].weight.data[deconv_blk._modules['deconv1'].weight.data<0] = 0
            deconv_blk._modules['deconv2'].weight.data = conv_blk._modules['conv1'].weight.data
            try:
                deconv_blk._modules['upsample'].weight.data = conv_blk._modules['downsample'][0].weight.data
                deconv_blk._modules['upsample'].weight.data[deconv_blk._modules['upsample'].weight.data<0] = 0
            except KeyError:
                pass
            #deconv_blk._modules['deconv2'].weight.data[deconv_blk._modules['deconv2'].weight.data<0] = 0
        else:
        # resize
            deconv_blk._modules['deconv1'][2].weight.data = conv_blk._modules['conv2'].weight.data.transpose(0,1).flip(2,3)
        # deconv_blk._modules['deconv1'][2].weight.data[deconv_blk._modules['deconv1'][2].weight.data<0] = 0
            deconv_blk._modules['deconv2'][2].weight.data = conv_blk._modules['conv1'].weight.data.transpose(0,1).flip(2,3)
            #deconv_blk._modules['deconv2'][2].weight.data[deconv_blk._modules['deconv2'][2].weight.data<0] = 0
            try:
                deconv_blk._modules['upsample'].weight.data = conv_blk._modules['downsample'][0].weight.data.transpose(0,1)
            except KeyError:
                pass
if deconv_resnet.name == 'transposeConv':    
    deconv_layers['conv_last'].weight.data = conv_layers['conv1'].weight.data
    deconv_layers['conv_last'].weight.data[deconv_layers['conv_last'].weight.data<0] = 0
else:
#resize       
    deconv_layers['conv_last'][2].weight.data = conv_layers['conv1'].weight.data.transpose(0,1).flip(2,3)
    #deconv_layers['conv_last'][2].weight.data[deconv_layers['conv_last'][2].weight.data<0] = 0


# visualize
# layer : 1 - 4
layer = 3
block = 1
num_maps = 9
actv_sele,visual_results = visualize(layer,block,activations,deconv_resnet,num_maps = num_maps)

n_plts = int(np.sqrt(num_maps))
#plt.ion()
end =  time.time()
print('With deconv, time used is : ', end-start,'s')

def enhance(img):
    img_copy = img.copy()
    #median = np.zeros(3)
    img_filtered = gaussian(img_copy,multichannel=True)
    
    return img_filtered

#fig1, ax1= plt.subplots(n_plts, n_plts)
fig2, ax2= plt.subplots(n_plts, n_plts)
#fig3, ax3= plt.subplots(n_plts, n_plts)
  
for row in range(n_plts):
    for col in range(n_plts):
       # ax1[row,col].imshow(actv_sele[row*3+col],cmap='gray')
        #ax1[row,col].axis('off')
        ax2[row,col].imshow(enhance(visual_results[row*3+col][0]))
        #coordinates = visual_results[row*3+col][1]
        #ax3[row,col].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax2[row,col].axis('off')
#fig1.suptitle('activations: layer{}, block{}'.format(layer,block))
fig2.suptitle('visualizations: layer{}, block{}'.format(layer,block))
plt.show(block=False)
plt.show()

#fig1.savefig('figs/deconv_layer{}_block{}_activations.png'.format(layer,block))
fig2.savefig('figs/deconv_layer{}_block{}_visualization.png'.format(layer,block))