import matplotlib.pyplot as plt

from math import sqrt, ceil
import torch
import numpy as np
import cv2
import pdb
from skimage import exposure, img_as_ubyte
from skimage.feature import peak_local_max,corner_peaks

from skimage.filters.rank import median
from skimage.morphology import disk
# visualize a feature map in a grid
def vis_grid(feat_map): # feat_map: (C, H, W, 1)
    (C, H, W, B) = feat_map.shape
    cnt = int(ceil(sqrt(C)))
    G = np.ones((cnt * H + cnt, cnt * W + cnt, B), feat_map.dtype)  # additional cnt for black cutting-lines
    G *= np.min(feat_map)

    n = 0
    for row in range(cnt):
        for col in range(cnt):
            if n < C:
                # additional cnt for black cutting-lines
                G[row * H + row : (row + 1) * H + row, col * W + col : (col + 1) * W + col, :] = feat_map[n, :, :, :]
                n += 1

    # normalize to [0, 1]
    G = (G - G.min()) / (G.max() - G.min())

    return G

# visualize a layer (a feature map represented by a grid)
def vis_layer(feat_map_grid):
    plt.clf()   # clear figure
    plt.subplot(121)
    plt.imshow(feat_map_grid[:, :, 0], cmap = 'gray')   # feat_map_grid: (ceil(sqrt(C)) * H, ceil(sqrt(C)) * W, 1)

# transform and normalize a deconvolutional output image
def tn_deconv_img(deconv_output):
    #print(deconv_output[0].shape)
    img_ori = deconv_output.data[0].permute(1, 2, 0).numpy()  # (H, W, C)

    img = img_ori.copy()
    
    img = img - img.min()
    img /= img.max()

    return img