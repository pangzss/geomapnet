import torch
from utils import *
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
def patches_grid(patches): # feat_map: (C, H, W, 1)
    # input patch : 9x3x50x50 or 9x3x100x100
    patches = patches.transpose(0,2,3,1)
    (B,H,W,C) = patches.shape
    cnt = 8
    G = np.ones((cnt * H + cnt, cnt * W + cnt+2, C), patches.dtype)  # additional cnt for black cutting-lines
    G *= np.min(patches)

    n = 0
    for row in range(cnt):
        for col in range(cnt):
            if n < B:
                # additional cnt for black cutting-lines
                G[row * H + row : (row + 1) * H + row, col * W + col : (col + 1) * W + col, :] = patches[n, :, :, :]
                n += 1

    # normalize to [0, 1]
    G = G - G.min()
    if G.max() != 0 :
        G /= G.max() 

    return G
G_all = []
for style in [0,4,8,16]:

    weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
                4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
                8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
                16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
    weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[style])
    model = get_model(weights_dir)

    weights = model._modules['conv1'].weight.data.numpy()
    for i in range(weights.shape[0]):
        weights[i] = norm_std(weights[i])

    if style == 0:
        G = patches_grid(weights)
    else:
        G = np.concatenate((G,patches_grid(weights)),axis = 1)

plt.imshow(G[:,:-5,:])
plt.axis('off')
plt.show()
