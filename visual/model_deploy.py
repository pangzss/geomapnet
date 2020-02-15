import torch
from conv_resnet import resnet34
from deconv_resnet import Deconv_ResNet
from feature_extraction import conv_forward
import matplotlib.pyplot as plt
# import mapnet 
import set_paths
from models.posenet import PoseNet, MapNet
#from torchvision import models 
import configparser

settings = configparser.ConfigParser()
with open('../scripts/configs/style.ini','r') as f:
    settings.read_file(f)
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
# adapted resnet34 with forward hooks
feature_extractor = resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)

mapnet_model = MapNet(mapnet=posenet)
# load weights
from common.train import load_state_dict
loc_func = lambda storage, loc: storage
weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'
checkpoint= torch.load(weights_dir,map_location=loc_func)
load_state_dict(mapnet_model,checkpoint['model_state_dict'])
#
conv_resnet = mapnet_model._modules['mapnet']._modules['feature_extractor']

fwd = conv_forward()
img = torch.ones((1,3,256,256))
activations,identity_maps = fwd.get_conv_maps(conv_resnet,img)
