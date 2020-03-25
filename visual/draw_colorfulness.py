import os
import os.path as osp
import matplotlib.pyplot as plt 
import numpy as np


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
    
dataset = 'AachenNight'
root_folder = osp.join('./figs',dataset+'_files')
mode_folder = osp.join(root_folder,'visualize_layer_stats')

num_blocks = [3,4,6,3]

for layer in range(1,4+1):
    for block in range(0,num_blocks[layer-1]):
        stats_path = osp.join(mode_folder,'layer_{}_block_{}'.format(layer,block)+'_colness.txt')
        with open(stats_path, 'r') as f:
                    stats_dict = eval(f.read())
        # building
        fig,ax = plt.subplots(1,1)
    
        stats = []
        for num_styles in [0,4,8,16]:
        
            stat = stats_dict[num_styles]
        
            stats.append(sorted(np.array(stats_dict[num_styles])))

        #axes[int(i>=2),i%2].hist(stat,10,alpha=0.5,label='{} style'.format(num_styles))
        parts = ax.violinplot(stats,showmeans=False, showmedians=False,
            showextrema=False)

        for pc in parts['bodies']:
            pc.set_facecolor('#3a82d4')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        
        quartile1, medians, quartile3 = np.percentile(stats, [25, 50, 75],axis=1)
        whiskers = np.array([\
            adjacent_values(sorted_array, q1, q3)\
            for sorted_array, q1, q3 in zip(stats, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
        #axes[int(i>=2),i%2].set_xlabel('gradient')
        ax.set_ylabel('Hue variation')
        set_axis_style(ax,[0,4,8,16])

        # axes[int(i>=2),i%2].legend(loc='upper right')
        fig.suptitle("Layer {} Block {}".format(layer,block))
        plt.subplots_adjust(wspace=1,hspace=0.5)

        fig.savefig(osp.join(mode_folder,'hue_violin_layer_{}_block_{}.jpg'.format(layer,block)))

        plt.close()