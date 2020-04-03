"""
Create file in format for submission on www.visuallocalization.net 
"""
from __future__ import division
import pickle
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torch.cuda
import configparser
import transforms3d.quaternions as txq
import set_paths
from models.posenet import *
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp, calc_vos_safe_fc, calc_vos_safe
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
from dataset_loaders.utils import load_image
import tqdm

parser = argparse.ArgumentParser(description='Create submission file for AachenDayNight dataset on www.visuallocalization.net')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar','AachenDayNight',
                                                'CambridgeLandmarks','stylized'),
                    help='Dataset')
parser.add_argument('--dir', type=str, required=True,help='Give directory where images to evaluate are')
parser.add_argument('--model', choices=('posenet', 'mapnet'), help='Model to use')
parser.add_argument('--weights', type=str, help='trained weights to load', required=True)
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--output_file', type=str, default='Aachen_query.txt', help='File to store output')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
train_split = section.getint('train_split', 6)
if (args.model.find('mapnet') >= 0):
    steps = section.getint('steps')
    skip = section.getint('skip')
    real = section.getboolean('real')
    variable_skip = section.getboolean('variable_skip')
    fc_vos = False

data_dir = osp.join('..', 'data', args.dataset)

filename = 'stats'

stats_file = osp.join(data_dir, '{:s}.txt'.format(filename))
print('Using {} as stats file'.format(stats_file))
stats = np.loadtxt(stats_file)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
resize = int(max(crop_size))


# model
feature_extractor = models.resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
if (args.model.find('mapnet') >= 0):
  model = MapNet(mapnet=posenet)
else:
  model = posenet
model.eval()

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
  def loc_func(storage, loc): return storage
  checkpoint = torch.load(weights_filename, map_location=loc_func)
  load_state_dict(model, checkpoint['model_state_dict'])
  print('Loaded weights from {:s}'.format(weights_filename))
else:
  print('Could not load weights from {:s}'.format(weights_filename))
  sys.exit(-1)
  
# transformer
data_transform = transforms.Compose([
  transforms.Resize(resize),
  transforms.CenterCrop(crop_size),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
print(data_transform)
# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir,'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
print('Load stats from %s\npose mean: %s\npose std: %s'%(data_dir,str(pose_m), str(pose_s)))

files = []
for dirpath, dirnames, filenames in os.walk(args.dir):
    files += [(f, os.path.join(dirpath, f)) for f in filenames if f.endswith('.jpg')]
print('Found %d files'%len(files))


# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    model.cuda()

L = len(files)
pred_poses = np.zeros((L, 7))  # store all predicted poses

for i, file in tqdm.tqdm(enumerate(files), total=L):
    #if i % 200 == 0:
    #    print('Image {:d} / {:d}'.format(i, len(files)))
    
    #print('Load file %s'%file)
    data = load_image(file[1])
    if args.model == 'mapnet':
      data = data_transform(data).unsqueeze(0).unsqueeze(0)
    else:
      data = data_transform(data).unsqueeze(0)
    _, output = step_feedfwd(data, model, CUDA, train=False)
    output = output[0]
    
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    
    pred_poses[i, :] = output[len(output) // 2]
    
    
        
output_file = open('logs/'+args.output_file, 'w')
for i, file in enumerate(files):
    filename = file[0]
    pose = pred_poses[i]
    q = pose[3:]
    p = pose[:3]
    t = -txq.rotate_vector(p, q)
    file_string = filename
    for f in q:
        file_string += ' '+str(f)
    for f in t:
        file_string += ' '+str(f)
    output_file.write(file_string+'\n')
output_file.close()
        

