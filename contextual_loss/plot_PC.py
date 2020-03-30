import sys
sys.path.append('../')
import os
import os.path as osp
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from common.utils import *

import cv2

from PIL import ImageDraw
from PIL import ImageFont

from CX_distance import CX_loss

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('#styles')


num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        # define root folder
        dataset = 'AachenPairs'
        root_folder = osp.join('./figs',dataset+'_files')
        mode_folder = osp.join(root_folder,'perceptual_loss_rand_pairs')

        with open(osp.join(mode_folder,'layer_{}_block_{}_pc_loss.txt'.format(layer,block)), 'r') as f:
                cx_loss_dict = eval(f.read())
        to_plot = []
        for num_style in [0,4,8,16]:

            to_plot.append(cx_loss_dict[num_style])
                

        fig,ax = plt.subplots(1,1)
        fig_8,ax_8 =  plt.subplots(1,1)
        #axes[int(i>=2),i%2].hist(stat,10,alpha=0.5,label='{} style'.format(num_styles))
        x = np.arange(len(to_plot[0]))
        ax.plot(x,to_plot[0],label='0 style')
        ax.plot(x,to_plot[1],label='4 style')
        ax.plot(x,to_plot[2],label='8 style')
        ax.plot(x,to_plot[3],label='16 style')
        ax.set_xlabel('pair index')
        ax.set_ylabel('perceptual loss')
        ax.legend(loc='upper right')
        fig.suptitle("Layer {} Block {}".format(layer,block))
        fig.savefig(osp.join(mode_folder,'PC_layer_{}_block_{}.jpg'.format(layer,block)))
        #ax_8.plot(x,to_plot[2],label='8 style')
        #ax_8.set_xlabel('pair index')
        #ax_8.set_ylabel('perceptual loss')
        #plt.legend(loc='upper right')
        #fig_8.suptitle("Layer {} Block {}".format(layer,block))

        #ax_np =  get_img_from_fig(fig)
        #ax_np = np.uint8((ax_np - ax_np.min())/(ax_np.max()-ax_np.min())*255)

        #ax_np_8 =  get_img_from_fig(fig_8)
        #ax_np_8 = np.uint8((ax_np_8 - ax_np_8.min())/(ax_np_8.max()-ax_np_8.min())*255)

        #to_save = np.concatenate((ax_np,np.uint8(np.ones((ax_np.shape[0],3,3))*255),ax_np_8),axis=1)
        #file_name_to_export = 'PC_layer_'+str(layer)+'_block_'+str(block)
        #to_folder = osp.join(mode_folder)

        #save_original_images(to_save, to_folder, file_name_to_export)


        # no outliers
        for i in range(len(to_plot)):
            pc = to_plot[i]
            median = np.median(np.asarray(pc))
            pc = [pc[k] if pc[k]<median else median for k in range(len(pc) )]
            to_plot[i] = pc
        fig,ax = plt.subplots(1,1)
        #axes[int(i>=2),i%2].hist(stat,10,alpha=0.5,label='{} style'.format(num_styles))
        x = np.arange(len(to_plot[0]))
        ax.plot(x,to_plot[0],label='0 style')
        ax.plot(x,to_plot[1],label='4 style')
        ax.plot(x,to_plot[2],label='8 style')
        ax.plot(x,to_plot[3],label='16 style')
        ax.set_xlabel('pair index')
        ax.set_ylabel('perceptual loss')
        ax.legend(loc='upper right')
        fig.suptitle("Layer {} Block {}".format(layer,block))

        fig.savefig(osp.join(mode_folder,'PC_layer_{}_block_{}_no_outliers.jpg'.format(layer,block)))