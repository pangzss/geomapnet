import sys
sys.path.append('../')

import os
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
    h,w = max_index

    radius = (kernel_size - 1)/2

    left = w - radius
    right = w + radius + 1
    w_slice = slice( int(left*(left>=0) - (right-W+1)*(right>=W)), 
                     int(right - (right-W+1)*(right>=W) - left*(left<0)))

    up = h - radius

    down = h + radius + 1

    h_slice = slice(int(up*(up>=0) -(down-H+1)*(down>=H)), 
                    int(down - (down-H+1)*(down>=H) - up*(up<0)))

    return (h_slice, w_slice)
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
    gradient = gradient - gradient.min()
    if gradient.max() != 0:
        gradient /= gradient.max()
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