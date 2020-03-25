import sys
sys.path.append('../')
import os.path as osp
import numpy as np 
from torchvision import models
from common.utils import *

style = 0
weights_name = {0:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_baseline.pth.tar',
            4: 'AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar',
            8:'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_8_styles_seed0.pth.tar',
            16: 'AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar'}       
weights_dir = osp.join('../scripts/logs/stylized_models',weights_name[style])
model = get_model(weights_dir,return_mapnet=True)

weights_xyz = model._modules['mapnet']._modules['fc_xyz'].weight.data.numpy()

