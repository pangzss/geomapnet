import sys
sys.path.append('../')

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
start = time.time()


def preprocess(img):
    img = np.asarray(img.resize((512, 512))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img_mean = np.mean(img)
    img_std  = np.std(img)
    img = (img - img_mean)/img_std
    img = img.transpose(2, 0, 1)    # reshape to (C, H, W)
    img = img[np.newaxis, :, :, :]  # add one dim to (1, C, H, W)
    
    return Variable(torch.FloatTensor(img.astype(float)))
 

def load_image(filename):
    img = Image.open(filename)
    return img


def visualize(layer,block,activations,deconv_model,num_maps = 1):
    assert layer>=1 and layer<=4 , "layer index must range from 1 to 4"
    actv_maps_sele = []
    deconv_imgs = []
    num_blocks_conv = [3,4,6,3]
    num_blocks_curr_layer = num_blocks_conv[layer-1]
    #for block in range(num_blocks_curr_layer):
    actv_maps = activations['layer'+str(layer)]['block'+str(block)]

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

        img,img_ori = tn_deconv_img(deconv_output)
        
        actv_maps_sele.append(actv_maps_copy.data.numpy()[0][index])
        deconv_imgs.append([img,img_ori])

    return actv_maps_sele, deconv_imgs            


#config 
parser = argparser.ArgumentParser(description='Filter visualization for MapNet')
parser.add_argument('--dataset', type=str, choices=('7Scenes','AachenDay','AachenNight'
                                                    'Cambridge','stylized'),
                    help = 'Dataset')
parser.add_argument('--model',type=str,choices=('stylized','ordinary'))
parser.add_argument('--weights', type=str, help='trained weights to load')
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
weights_dir = './logs' + args.weights
checkpoint= torch.load(weights_dir,map_location=loc_func)
load_state_dict(mapnet_model,checkpoint['model_state_dict'])


####### visualization #######
#img_file = './aachen1.jpg'
path = 'data/'+args.dataset+'/'
img_dirs = os.listdir(path) 

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
#for layer in range(1,4+1):
#    conv_layer = conv_layers['layer'+str(layer)]
#    deconv_layer = deconv_layers['layer'+str(4-layer+1)]
#    for blk in range(0,num_blocks_conv[layer-1]):
#        conv_blk = conv_layer[blk]
#        deconv_blk = deconv_layer[num_blocks_conv[layer-1]-blk-1]
#         
#        deconv_blk._modules['deconv1'].weight.data = conv_blk._modules['conv2'].weight.data
#        deconv_blk._modules['deconv2'].weight.data = conv_blk._modules['conv1'].weight.data
        #deconv_blk._modules['bn1'].weight.data = conv_blk._modules['bn2'].weight.data
        #deconv_blk._modules['bn2'].weight.data = conv_blk._modules['bn1'].weight.data
#deconv_layers['conv_last'].weight.data = conv_layers['conv1'].weight.data
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
# visualize
# layer : 1 - 4
# block : 3 4 6 3
layer = 3
block = 4
num_maps = 9
actv_sele,visual_results = visualize(layer,block,activations,deconv_resnet,num_maps = num_maps)

n_plts = int(np.sqrt(num_maps))
#plt.ion()

end = time.time()
print('With resize, time used is : ', end-start,'s')

def enhance(img):
    img_copy = img.copy()
    median = np.zeros(3)
    for i in range(3):
        median[i] = np.median(img[:,:,i])
    img_copy[img_copy <= 1.3*median] = 0
    return img_copy

fig1, ax1= plt.subplots(n_plts, n_plts)
plt.subplots_adjust(wspace=0,hspace=0.05)
fig2, ax2= plt.subplots(n_plts, n_plts)
plt.subplots_adjust(wspace=0,hspace=0.05)
for row in range(n_plts):
    for col in range(n_plts):
        ax1[row,col].imshow(actv_sele[row*3+col],cmap='gray')
        ax1[row,col].axis('off')
        ax2[row,col].imshow(enhance(visual_results[row*3+col][0]))
        ax2[row,col].axis('off')
fig1.suptitle('activations: layer{}, block{}'.format(layer,block))
fig2.suptitle('visualizations: layer{}, block{}'.format(layer,block))

plt.show(block=False)
plt.show()

fig1.savefig('figs/resize_layer{}_block{}_activations.png'.format(layer,block))
fig2.savefig('figs/resize_layer{}_block{}_visualization.png'.format(layer,block))