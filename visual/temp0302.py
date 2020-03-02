import matplotlib.pyplot as plt
import numpy as np 

def patches_grid(patches): # feat_map: (C, H, W, 1)
    # input patch : 9x3x50x50 or 9x3x100x100
    patches = patches.transpose(0,2,3,1)
    (B,H,W,C) = patches.shape
    cnt = 3
    G = np.ones((cnt * H + cnt+5, cnt * W + cnt+5 , C), patches.dtype)  # additional cnt for black cutting-lines
    G *= np.min(patches)

    n = 0
    for row in range(cnt):
        for col in range(cnt):
            if n < B:
                # additional cnt for black cutting-lines
                G[row * H + row : (row + 1) * H + row, col * W + col : (col + 1) * W + col, :] = patches[n, :, :, :]
                n += 1

    # normalize to [0, 1]
    G = (G - G.min()) / (G.max() - G.min())

    return G

patches = np.random.randn(9,3,50,50)

G1 = patches_grid(patches)
G2 = patches_grid(patches)
G = np.concatenate((G1,G2),axis =1)
G = np.concatenate((G,G2),axis =1)
G = np.concatenate((G,G2),axis =1)
fig,ax = plt.subplots(ncols=1)
ax.imshow(G)   # feat_map_grid: (ceil(sqrt(C)) * H, ceil(sqrt(C)) * W, 1)
plt.axis('off')
plt.show()