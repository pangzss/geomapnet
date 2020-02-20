import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import numpy as np

from conv_resnet import resnet34
from resize_resnet import Resize_ResNet
from feature_extraction import conv_forward
from utils import *

import cv2
from skimage import exposure
from common.train import load_state_dict
# import mapnet 
#import set_paths
from models.posenet import PoseNet, MapNet
#from torchvision import models 
import configparser

def preprocess(img):
    img = np.asarray(img.resize((512, 512))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img = img.transpose(2, 0, 1)    # reshape to (C, H, W)
    img = img[np.newaxis, :, :, :]  # add one dim to (1, C, H, W)
    return Variable(torch.FloatTensor(img.astype(float)))
 

def load_image(filename):
    img = Image.open(filename)
    return img


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
weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
checkpoint= torch.load(weights_dir,map_location=loc_func)
load_state_dict(mapnet_model,checkpoint['model_state_dict'])


####### visualization #######
#img_file = './aachen1.jpg'
img_file = './aachen1.jpg'
# load an image
img = load_image(img_file)
#plt.figure(figsize = (10, 5))
#imgplot = plt.imshow(img)
#plt.show(block=False)
#plt.show()
# preprocess an image, return a pytorch variable
input_img = preprocess(img)
# define conv model
conv_resnet = mapnet_model._modules['mapnet']._modules['feature_extractor']
conv_resnet.eval()
# collect activations
fwd = conv_forward()
out,activations,identity_maps = fwd.get_conv_maps(conv_resnet,input_img)
# define deconv model
deconv_resnet = Resize_ResNet(identity_maps)
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
        
        deconv_blk._modules['deconv1'][2].weight.data = conv_blk._modules['conv2'].weight.data.flip(2,3).transpose(0,1)
        deconv_blk._modules['deconv2'][2].weight.data = conv_blk._modules['conv1'].weight.data.flip(2,3).transpose(0,1)
        #deconv_blk._modules['bn1'].weight.data = conv_blk._modules['bn2'].weight.data
        #deconv_blk._modules['bn2'].weight.data = conv_blk._modules['bn1'].weight.data
deconv_layers['conv_last'][2].weight.data = conv_layers['conv1'].weight.data.flip(2,3).transpose(0,1)

plt.ion()

plt.figure(figsize=(10,5))

while True:
    layer = input('which layer to view (1-4), anything else to exit):')
    try:
        layer = int(layer)
    except ValueError:
        continue
    if (isinstance(layer,int) is False) or layer < 1 or layer >4:
        sys.exit(0)
    num_blocks_curr_layer = num_blocks_conv[layer-1]
    block = input('which block to view (0~{}), anything else to exit):'.format(num_blocks_curr_layer-1))
    try:
        block = int(block)
    except ValueError:
        continue
    if (isinstance(block,int) is False) or block < 0 or block > num_blocks_curr_layer:
        sys.exit(0)

    actv_map = activations['layer'+str(layer)]['block'+str(block)]
    actv_map_grid = vis_grid(actv_map.data.numpy().transpose(1, 2, 3, 0))
    vis_layer(actv_map_grid)

    n_activation = actv_map.shape[1]

    marker = None
    while True:
      
        select = input('Select an activation? [y/n]: ') == 'y'
        if marker != None:
            marker.pop(0).remove()
        if not select:
            break
    
        _,_,H,W = actv_map.shape
        gridH, gridW, _ = actv_map_grid.shape

        col_steps = gridW // (W + 1)

        print('Click on an activation to continue')
        xPos, yPos = plt.ginput(1)[0]
        xIdx = xPos // (W + 1)  # additional "1" represents black cutting-line
        yIdx = yPos // (H + 1)
        activation_idx = int(col_steps * yIdx + xIdx)   # index starts from 0

        if activation_idx >= n_activation:
                print('Invalid activation selected!')
                continue
        

        actv_map_copy = actv_map.clone()
        
        if activation_idx == 0:
            actv_map_copy[:,1:,:,:] = 0
        else:
            actv_map_copy[:,:activation_idx,:,:] = 0
            if activation_idx != actv_map_copy.shape[1] - 1:
                actv_map_copy[:,activation_idx + 1:, :,:] = 0
        
        deconv_output =  deconv_resnet(actv_map_copy,layer,block)
        print('reach', deconv_output.shape)
        img,_ = tn_deconv_img(deconv_output)
        
        
        plt.subplot(121)
        marker = plt.plot(xPos, yPos, marker = '+', color = 'red')
        plt.subplot(122)
        #plt.imshow(heatmap) # img3
        plt.imshow(img)

