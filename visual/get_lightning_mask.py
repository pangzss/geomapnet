import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io
from skimage.transform import resize
import cv2
from utils import save_original_images
# Load image
dataset = 'AachenNight'
path = osp.join('data', dataset)
img_dirs = os.listdir(path)
for img_dir in img_dirs:
    img_path = osp.join(path,img_dir)
    
    img = io.imread(img_path)
    img_grey = cv2.imread(img_path, 0)
    img = resize(img,(112,112),anti_aliasing=True,preserve_range=True).astype(np.uint8)
    img_grey = resize(img_grey,(112,112),anti_aliasing=True,preserve_range=True).astype(np.uint8)
    blurred = img_grey #cv2.GaussianBlur(img_grey, (11, 11), 0)
    #blurred = np.uint8((blurred - blurred.min())/(blurred.max()-blurred.min())*255)
    thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)[1]
    #thresh_rf = cv2.erode(thresh, None, iterations=2)
    #thresh_rf = cv2.dilate(thresh, None, iterations=2)

    output = np.concatenate((img,np.repeat(thresh[...,None],3,axis=2)),axis=1)

    to_folder = './figs/AachenNight_files/binary_masks'
    file_name_to_export = img_dir.split('.')[0]+'_mask'
    save_original_images(output, to_folder, file_name_to_export)