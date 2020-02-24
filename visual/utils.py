import sys
sys.path.append('../')

import torch
import torchvision.models as models
from common.train import load_state_dict
from models.posenet import PoseNet, MapNet



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