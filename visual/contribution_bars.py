import sys
#sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torchvision import models
import pickle
import cv2

dataset = 'AachenNight'
root_folder = osp.join('./figs',dataset+'_files')
mode_folder = osp.join(root_folder,'contribition_bars')


layer = 4
block = 2

row = 0
col = 0
fig, axs = plt.subplots(8, 4,sharey=True,sharex=True, tight_layout=True)
fig_, axs_ = plt.subplots(8, 1,sharey=True,sharex=True, tight_layout=True)
for top_i in range(8):
    across_styles = []
    for num_styles in [0,4,8,16]:
        lists_folder = osp.join(root_folder,'visualize_layer','grads_lists_layer_{}'.format(layer))
        lists_name = 'grads_list_top_{}_layer_{}_block_{}_style_{}.txt'.format(top_i+1,layer,block,num_styles)

        with open(osp.join(lists_folder,lists_name),'rb') as f:
            grads_list = pickle.load(f)

        grads_array = np.stack(grads_list,axis=0)
        # grads_array contains *num_filters of gradient images. 
        # need to find each image's proportion to the whole

        grads_sum = np.clip(grads_array,0,grads_array.max()).sum(axis=(3,2,1))
        prop = grads_sum/grads_sum.sum()
        
        across_styles.append(grads_sum.sum())
        x = np.arange(len(prop)) 
        axs[row,col].stem(x,prop,use_line_collection=True, markerfmt=',')
        if col == 0:
            axs[row,col].set_ylabel('Proportion')
        if row == 7:
            axs[row,col].set_xlabel('Filter Index')
        if row == 0:
            axs[row,col].set_title('{} styles'.format(num_styles))
        #print(i,j)
        if col == 3:
            col = -1
            row += 1
        col += 1
    
    across_styles = np.stack(across_styles)
    across_styles = np.clip(across_styles,0,across_styles.max())
  

    x_ = np.arange(len(across_styles))
    weights = 0.35
    axs_[top_i].bar(x_,across_styles/across_styles.sum(),weights)
    if top_i == 0:
        axs_[top_i].set_title('gradient ratio')
    if top_i == 7:
        axs_[top_i].set_xticks(x_)
        axs_[top_i].set_xticklabels(['style0','style4','style8','style16'])
    axs_[top_i].set_ylabel('ratio')
    #axs_[top_i].set_ylim(0,1)
#fig.suptitle('Layer {}, Block {}'.format(layer,block))
#fig_.suptitle('Layer {}, Block {}'.format(layer,block))
plt.subplots_adjust(wspace=1,hspace=0.5)
plt.show()