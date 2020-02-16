import matplotlib.pyplot as plt

from math import sqrt, ceil
import numpy as np
import cv2
import pdb
from skimage import exposure
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
    img_ori = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    img = img_ori.copy()
    #equ = cv2.equalizeHist(img)

    # normalize
    for i in range(3):
        if img[:,:,i].max() != img[:,:,i].min():
            img[:,:,i] = 255*(img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max() - img[:,:,i].min() )
        else:
            img[:,:,i] = 255*(img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max()+1e-05)
    #img[:,:,1] = (img[:,:,1] - img[:,:,1].min()) / (img[:,:,1].max() - img[:,:,1].min())*255
    #img[:,:,2] = (img[:,:,2] - img[:,:,2].min()) / (img[:,:,2].max() - img[:,:,2].min())*255
    img = img.astype(np.uint8)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #img = exposure.rescale_intensity(img)
    img = exposure.adjust_gamma(img, 2)
   
    return img,img_ori