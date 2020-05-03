"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Main training script for MapNet
"""
import set_paths
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion, TripletCriterion
from models.posenet import PoseNet, MapNet, TriNet, StripNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
import random
parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar','AachenDayNight','Cambridge'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default = ' ', help='Scene name')
parser.add_argument('--style_dir', type=str, help='the directory of style images')
parser.add_argument('--real_prob', type=int, help='the prob of using real images')
parser.add_argument('--alpha', type=float, help='intensity of stylization')
#parser.add_argument('--num_styles', type=int, help='number of styles')
parser.add_argument('--t_aug', action='store_true', help='use traditional augmentation')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++','trinet','stripnet'),
  help='Model to train')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of translation loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of rotation loss')
parser.add_argument('--learn_sigma', action='store_true',
  help='Learn the weights of contextual and pose losses')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
parser.add_argument('--init_seed', type=int, default=0, help='Set seed for random initialization of model')
parser.add_argument('--server', type=str, default='http://localhost', help='Set visdom server address')
parser.add_argument('--port', type=int, default=8097, help='set visdom port')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in list(section.items()) if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)

sc = section.getfloat('sigma_cx', 0.0)
sp = section.getfloat('sigma_pc',0.0)

sax = section.getfloat('beta_translation', 0.0)
saq = section.getfloat('beta')
train_split = section.getint('train_split', 6)
if args.model.find('mapnet') >= 0 or args.model=='trinet' or args.model == 'stripnet':
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = section.getfloat('gamma_translation', 0.0)
  srq = section.getfloat('gamma')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print('Using {:s} VO'.format(vo_lib))

section = settings['training']
min_perceptual = section.getboolean('min_perceptual',False)
#seed = section.getint('seed')
seed = args.init_seed
if seed >= 0:
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# model
feature_extractor = models.resnet34(pretrained=True)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model=='mapnet++'))
if args.model == 'posenet':
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
elif args.model == 'trinet':
  model = TriNet(trinet=posenet)
elif args.model == 'stripnet':
  model = StripNet(stripnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model == 'posenet':
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
  if args.model.find('++') >= 0:
    kwargs = dict(kwargs, gps_mode=(vo_lib=='gps') )
    train_criterion = MapNetOnlineCriterion(**kwargs)
    val_criterion = MapNetOnlineCriterion()
  else:
    kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
    train_criterion = MapNetCriterion(**kwargs)
    val_criterion = MapNetCriterion()
elif args.model == 'trinet':
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, sc=sc, sp=sp,learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma, learn_sigma=args.learn_sigma)
  train_criterion = TripletCriterion(**kwargs)
  val_criterion = TripletCriterion()
elif args.model == 'stripnet':
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, sc=sc, sp=sp,learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma, learn_sigma=args.learn_sigma)
  train_criterion = TripletCriterion(**kwargs)
  val_criterion = TripletCriterion()
else:
  raise NotImplementedError

# optimizer
param_list = [{'params': model.parameters()}]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
if args.learn_sigma and hasattr(train_criterion, 'sc'):
  param_list.append({'params': [train_criterion.sc]})
if args.learn_sigma and hasattr(train_criterion, 'sp'):
  param_list.append({'params': [train_criterion.sp]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, **optim_config)

data_dir = osp.join('..', 'data', args.dataset)
if args.dataset == '7Scenes' or args.dataset =='Cambridge':
  stats_file = osp.join(data_dir, args.scene, 'stats.txt')
else:
  stats_file = osp.join(data_dir, 'stats.txt')
stats = np.loadtxt(stats_file)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
resize = int(max(crop_size))
# transformers
if args.dataset == 'AachenDayNight':
    tforms = [transforms.Resize(resize)]
    tforms.append(transforms.CenterCrop(crop_size))
else:
    tforms = [transforms.Resize(resize)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  print('Using ColorJitter data augmentation')
  tforms.append(transforms.ColorJitter(brightness=color_jitter,
    contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
# val transformers
if args.dataset == 'AachenDayNight':
    val_tforms = [transforms.Resize(resize)]
    val_tforms.append(transforms.CenterCrop(crop_size))
else:
    val_tforms = [transforms.Resize(resize)]
val_tforms.append(transforms.ToTensor())
val_tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
val_data_transform = transforms.Compose(val_tforms)
# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
  target_transform=target_transform, seed=seed)
val_kwargs = dict(scene=args.scene, data_path=data_dir, transform=val_data_transform,
  target_transform=target_transform, seed=seed)
if args.model == 'posenet':
  if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    train_set = SevenScenes(train=True, **kwargs)
    val_set = SevenScenes(train=False, **val_kwargs)
  elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    train_set = RobotCar(train=True, **kwargs)
    val_set = RobotCar(train=False, **val_kwargs)
  elif args.dataset == 'AachenDayNight':
    kwargs = dict(kwargs, real_prob=args.real_prob, style_dir = args.style_dir)
    from dataset_loaders.aachen_day_night import AachenDayNight
    train_set = AachenDayNight(train=True, **kwargs)
    val_set = AachenDayNight(train=False, **val_kwargs)
  
    
  elif args.dataset == 'Cambridge':
    kwargs = dict(kwargs,real_prob=args.real_prob, style_dir = args.style_dir)
    from dataset_loaders.cambridge import Cambridge
    train_set = Cambridge(train=True, **kwargs)
    val_set = Cambridge(train=False, **val_kwargs)
  
  else:
    raise NotImplementedError
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  val_kwargs = dict(val_kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  if args.model.find('++') >= 0:
    train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib=='gps'), **kwargs)
    val_set = None
  else:
    kwargs = dict(kwargs, real_prob=args.real_prob, style_dir = args.style_dir)
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **val_kwargs)
elif args.model == 'trinet':
  if args.dataset == 'AachenDayNight':
    from dataset_loaders.aachen_triplet import AachenTriplet
    kwargs = dict(kwargs, real_prob=args.real_prob, style_dir = args.style_dir)
    train_set = AachenTriplet(train=True, **kwargs)
    val_set = AachenTriplet(train=False, **val_kwargs)
  elif args.dataset == 'Cambridge':
    from dataset_loaders.cambridge_triplet import CambridgeTriplet
    kwargs = dict(kwargs, real_prob=args.real_prob, style_dir = args.style_dir,min_perceptual=min_perceptual)
    train_set = CambridgeTriplet(train=True, **kwargs)
    val_set = CambridgeTriplet(train=False, **val_kwargs)
elif args.model == 'stripnet': 
  if args.dataset == 'Cambridge':
    from dataset_loaders.cambridge_stripnet import CambridgeStripnet
    kwargs = dict(kwargs, real_prob=args.real_prob, style_dir = args.style_dir)
    train_set = CambridgeStripnet(train=True, **kwargs)
    val_set = CambridgeStripnet(train=False, **val_kwargs)
else:
  raise NotImplementedError

# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
if args.dataset == '7Scenes' or args.dataset == 'Cambridge':
  experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
    args.model, config_name)
else:
  experiment_name = '{:s}_{:s}_{:s}'.format(args.dataset,
    args.model, config_name)

experiment_name = '{:s}_{}_percent_real'.format(experiment_name,args.real_prob)
if args.learn_beta:
  experiment_name = '{:s}_beta'.format(experiment_name)
if args.learn_gamma:
  experiment_name = '{:s}_gamma'.format(experiment_name)
if args.learn_sigma:
  experiment_name = '{:s}_sigma'.format(experiment_name)

if args.real_prob < 100:
  experiment_name = '{:s}_alpha{}'.format(experiment_name, args.alpha if args.alpha>=0 else 'Rand')
if args.t_aug:
  experiment_name = '{:s}_aug'.format(experiment_name)
if seed >= 0:
  experiment_name = '{:s}_seed{}'.format(experiment_name, seed) 
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, seed=seed, alpha=args.alpha,device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion,visdom_server = args.server, visdom_port = args.port)
lstm = args.model == 'vidloc'
trinet = args.model == 'trinet'
if args.dataset == 'AachenDayNight' or args.dataset =='Cambridge':
  trainer.style_train_val(trinet=trinet)
else:
  trainer.train_val(lstm=lstm)
