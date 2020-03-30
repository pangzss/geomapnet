"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Computes the mean and std of pixels in a dataset
"""
import os.path as osp
import numpy as np
import argparse
import set_paths
#from . import set_paths
from dataset_loaders.seven_scenes import SevenScenes
#from dataset_loaders.robotcar import RobotCar
from dataset_loaders.aachen_day_night import AachenDayNight
from torchvision import transforms
from torch.utils.data import DataLoader
from common.train import safe_collate

parser = argparse.ArgumentParser(description='Dataset images statistics')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar','AachenDayNight'),
                    help='Dataset', required=True)
parser.add_argument('--num_styles', type=int, default=0, help='number of styles per image')
parser.add_argument('--scene', type=str,help='Scene name', required=False)
args = parser.parse_args()

data_dir = osp.join('..', 'data', args.dataset)

crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.RandomCrop(crop_size),
  transforms.ToTensor()])


# dataset loader
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=True, real=False,
  transform=data_transform)
if args.dataset == '7Scenes':
  dset = SevenScenes(**kwargs)
#elif args.dataset == 'RobotCar':
#  dset = RobotCar(**kwargs)
elif args.dataset == 'AachenDayNight':
  kwargs = dict(kwargs,num_styles=args.num_styles)
  dset = AachenDayNight(**kwargs)
else:
  raise NotImplementedError

# accumulate
batch_size = 90
num_workers = 8
loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                    collate_fn=safe_collate)
      
acc = np.zeros((3, crop_size[0], crop_size[1]))
sq_acc = np.zeros((3, crop_size[0], crop_size[1]))
if args.dataset != 'AachenDayNight':
  for batch_idx, (imgs, _) in enumerate(loader):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs**2, axis=0)

    if batch_idx % 50 == 0:
      print('Accumulated {:d} / {:d}'.format(batch_idx*batch_size, len(dset)))
else:
  for batch_idx, (imgs, _) in enumerate(loader):
    real = imgs[0].numpy()
    if args.num_styles != 0:
      style = [imgs[1][i].numpy() for i in range(args.num_styles)]
      style = np.concatenate(style,axis=0)
      
      imgs = np.concatenate((real,style),axis=0)
    
    else:
      imgs = real
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs**2, axis=0)

    if batch_idx % 50 == 0:
      print('Accumulated {:d} / {:d}'.format(batch_idx*batch_size, len(dset)))

N = len(dset)*(args.num_styles+1) * acc.shape[1] * acc.shape[2]

mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
mean_p /= N
print('Mean pixel = ', mean_p)

# std = E[x^2] - E[x]^2
std_p = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std. pixel = ', std_p)

if args.dataset == '7Scenes':
  output_filename = osp.join('..', 'data', args.dataset, args.scene, 'stats.txt')
else:
  output_filename = osp.join('..', 'data', args.dataset, 'stats_{}_styles.txt'.format(args.num_styles))
np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')
print('{:s} written'.format(output_filename))
