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

#from CX_distance import CX_loss

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

for layer in range(4,4+1):
    for block in range(0,num_blocks[layer-1]):
        print(layer,block)
        # define root folder
        dataset = 'AachenPairs'
        root_folder = osp.join('./figs',dataset+'_files')
        mode_folder = osp.join(root_folder,'contextual_loss')

        with open(osp.join(mode_folder,'layer_{}_block_{}_cx_loss.txt'.format(layer,block)), 'r') as f:
                cx_loss_dict = eval(f.read())
        to_plot = []
        for num_style in [0,4,8,16]:
            to_plot.append(cx_loss_dict[num_style])
                

        fig,ax = plt.subplots(1,1)

        #axes[int(i>=2),i%2].hist(stat,10,alpha=0.5,label='{} style'.format(num_styles))
        parts = ax.violinplot(to_plot,showmeans=False, showmedians=False,
                    showextrema=False)

        for pc in parts['bodies']:
            pc.set_facecolor('#6b6380')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = np.percentile(to_plot, [25, 50, 75],axis=1)
        whiskers = np.array([\
            adjacent_values(sorted_array, q1, q3)\
            for sorted_array, q1, q3 in zip(to_plot, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
        #axes[int(i>=2),i%2].set_xlabel('gradient')
        ax.set_ylabel('Contextual loss')
        set_axis_style(ax,[0,4,8,16])

        # axes[int(i>=2),i%2].legend(loc='upper right')
        fig.suptitle("Layer {} Block {}".format(layer,block))
        plt.subplots_adjust(wspace=1,hspace=0.5)

        fig.savefig(osp.join(mode_folder,'CX_violin_layer_{}_block_{}.jpg'.format(layer,block)))

