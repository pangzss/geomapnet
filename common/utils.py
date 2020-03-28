import sys
sys.path.append('../')

import os
import io
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import cv2
import torch
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from common.train import load_state_dict
from models.posenet import PoseNet, MapNet
# get model

def get_model(weights_dir):
    # adapted resnet34 with forward hooks
    feature_extractor = models.resnet34(pretrained=False)
    posenet = PoseNet(feature_extractor, droprate=0., pretrained=False)

    mapnet_model = MapNet(mapnet=posenet)
    # load weights
    loc_func = lambda storage, loc: storage
    #weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
    #weights_dirs = ['./logs/stylized_models/AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
    #                '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar']
    checkpoint= torch.load(weights_dir,map_location=loc_func)
    load_state_dict(mapnet_model,checkpoint['model_state_dict'])

    feature_extractor = mapnet_model._modules['mapnet']._modules['feature_extractor']
    model = feature_extractor 
    return model
# get maxima patches
def get_patch_slice(img, max_index,kernel_size=50):

    
    H,W = img.shape[-2:]
    h,w = max_index[:]

    #radius_H = (RF[0] - 1)/2
    #radius_W = (RF[1] - 1)//2
    radius = ( kernel_size - 1 ) / 2

    left = w - radius
    right = w + radius + 1
    w_slice = slice( int(left*(left>=0) - (right-W+1)*(right>=W)), 
                     int(right - (right-W+1)*(right>=W) - left*(left<0)))

    up = h - radius

    down = h + radius + 1

    h_slice = slice(int(up*(up>=0) -(down-H+1)*(down>=H)), 
                    int(down - (down-H+1)*(down>=H) - up*(up<0)))

    return (h_slice, w_slice)

    def patches_grid(patches): # feat_map: (C, H, W, 1)
        # input patch : 9x3x50x50 or 9x3x100x100
        patches = patches.transpose(2,3,1)
        (B,H,W,C) = patches.shape
        cnt = 3
        G = np.zeros((cnt * H + cnt, cnt * W + cnt, C), patches.dtype)  # additional cnt for black cutting-lines
        

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
# process images
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
def save_original_images(img, path, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
   
    # Save image
    path_to_file = os.path.join(path, file_name + '.png')
    save_image(img, path_to_file)
def save_gradient_images(gradient, path, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """

    if not os.path.exists(path):
            os.makedirs(path)
    # Normalize
    #gradient = gradient - gradient.min()
    #if gradient.max() != 0:
    #    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(path, file_name + '.png')
    save_image(gradient, path_to_file)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
        
    im.save(path)

def preprocess(img):

    img = np.asarray(img.resize((224, 224))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)
   
    img = img[None,:,:,:]
    return img

def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img

def bounding_box(grads_to_save,img_to_save,patch_slice):
    grads_to_save[0,patch_slice[0],patch_slice[1].start] = grads_to_save.max()
    grads_to_save[1:,patch_slice[0],patch_slice[1].start] = grads_to_save.min()

    grads_to_save[0,patch_slice[0],patch_slice[1].stop] = grads_to_save.max()
    grads_to_save[1:,patch_slice[0],patch_slice[1].stop] = grads_to_save.min()

    grads_to_save[0,patch_slice[0].start,patch_slice[1]] = grads_to_save.max()
    grads_to_save[1:,patch_slice[0].start,patch_slice[1]] = grads_to_save.min()

    grads_to_save[0,patch_slice[0].stop,patch_slice[1]] = grads_to_save.max()
    grads_to_save[1:,patch_slice[0].stop,patch_slice[1]] = grads_to_save.min()
    
    img_to_save[patch_slice[0],patch_slice[1].start,0] = 255
    img_to_save[patch_slice[0],patch_slice[1].start,1:] = 0

    img_to_save[patch_slice[0],patch_slice[1].stop,0] = 255
    img_to_save[patch_slice[0],patch_slice[1].stop,1:] = 0

    img_to_save[patch_slice[0].start,patch_slice[1],0] = 255
    img_to_save[patch_slice[0].start,patch_slice[1],1:] = 0

    img_to_save[patch_slice[0].stop,patch_slice[1],0] = 255
    img_to_save[patch_slice[0].stop,patch_slice[1],1:] = 0

    return grads_to_save,img_to_save

def norm_std(img,scale=0.05):
    """ Normalization of conv2d filters for visualization
    https://github.com/jacobgil/keras-filter-visualization/blob/master/utils.py
    Args:
        filter_in: [size_x, size_y, n_channel]
    """
    x = img.copy()
    x -= x.mean()
    x /= (x.std() + 1e-5)
    # make most of the value between [-0.5, 0.5]
    x *= scale
    # move to [0, 1]
    x += 0.3
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb