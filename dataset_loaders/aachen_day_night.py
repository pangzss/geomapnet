from __future__ import division
import sys
from torch.utils import data
import numpy as np
import ntpath
import random
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import transforms3d.quaternions as txq
import os
import tqdm
from PIL import Image

sys.path.insert(0, '../')
from common.pose_utils import process_poses_quaternion
from common.utils import load_image
from common.vis_utils import show_batch, show_stereo_batch
from AdaIN import net
from AdaIN.function import adaptive_instance_normalization
available_styles = [
    'goeritz.jpg',
    'asheville.jpg',
    'mondrian.jpg',
    'scene_de_rue.jpg',
    'flower_of_life.jpg',
    'antimonocromatismo.jpg',
    'woman_with_hat_matisse.jpg',
    'trial.jpg',
    'sketch.png',
    'picasso_self_portrait.jpg',
    'picasso_seated_nude_hr.jpg',
    'la_muse.jpg',
    'contrast_of_forms.jpg',
    'brushstrokes.jpg',
    'the_resevoir_at_poitiers.jpg',
    'woman_in_peasant_dress.jpg'
]

num_styles = len(available_styles)



class Sample:
    def __init__(self, path, pose):
        self.path = path
        self.pose = pose
        self.stylized = None

class AachenDayNight(data.Dataset):
    def __init__(self, data_path, train,skip_images=False,real=False,transform=None, 
                target_transform=None, style_dir = None,real_prob = 100,
                train_split=20,seed=7,scene=None):
        np.random.seed(seed)
        self.data_path = data_path
        self.train = train
        #
        self.real_prob = real_prob if self.train else 100
        self.style_dir = style_dir if self.train else None
        self.available_styles = os.listdir(style_dir) if self.style_dir is not None else None
        print('real_prob: {}.\nstyle_dir: {}\nnum_styles: {}'.format(self.real_prob,self.style_dir,len(self.available_styles) \
                                                                                                if self.style_dir is not None else 0))
        #
        self.skip_images = skip_images
        self.train_split = train_split
        print('train status: {}. \ntrain&val ratio: {}'.\
              format(self.train,self.train_split))
        self.images = []
        self.poses = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        self.train_idces = []
        self.val_idces = []
        self.transform = transform
        self.target_transform = target_transform
    

        nvm_file_name = os.path.join(self.data_path,'3D_model','aachen_cvpr2018_db.nvm')
        nvm_file = open(nvm_file_name,'r')
        lines = nvm_file.readlines()
        num_pts = int(lines[2].strip())
        pose_lines = lines[3:num_pts+3]
        pose_lines = [x.strip().split(' ') for x in pose_lines]
        for i in tqdm.tqdm(range(len(pose_lines)), total=len(pose_lines),
                            desc='Read images and load pose', leave=False):
            pose_i = pose_lines[i]
            # rotation quaternion
            q = [float(x) for x in pose_i[2:6]]
            # camera center
            c = [float(x) for x in pose_i[6:9]]
            pose = np.asarray(c+q)
            self.poses.append(pose)
            #path = pose_i[0].split('/')
            #path = os.path.join(self.data_path, path[0],path[1])
            path = os.path.join(self.data_path,pose_i[0])
            self.images.append(Sample(path,pose))
            '''
            if self.num_styles != 0:
                img_dir, base_name = os.path.split(path)
                img_dir,_ = os.path.split(img_dir)
                styl_dir = os.path.join(img_dir, 'stylized')
                base_name = base_name.split('.')
                factor = len(available_styles) // self.num_styles
                #styl_list = []
                self.images[i].stylized = []
                for j in range(self.num_styles):
                    if self.num_styles == 1:
                        styl_name = base_name[0] + '_stylized_'+'asheville' + '.' + base_name[1]
                    else:
                        styl_name = base_name[0] + '_stylized_'+available_styles[(i+j*factor) % len(available_styles)] + '.' + base_name[1]
                    
                    styl_path = os.path.join(styl_dir, styl_name)
                    self.images[i].stylized.append(styl_path)
                
                    #styl_list.append(styl_path)
                #self.images[i].stylized = styl_list
            '''

        self.poses = np.vstack(self.poses)
         # generate pose stats file
        pose_stats_filename = os.path.join('../data/AachenDayNight','pose_stats.txt')

        if self.train:
            # optionally, use the ps dictionary to calc stats
            mean_t = self.poses[:, :3].mean(axis=0)
            std_t = self.poses[:, :3].std(axis=0)
            print('save and use pose stats file')
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
            print('use generated pose stats file')
        self.poses = process_poses_quaternion(self.poses[:,:3], self.poses[:,3:], mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
        # extract train or val data
        for i in tqdm.tqdm(range(len(self.images)), total=len(self.images), desc='Split dataset', leave=False):
            if i % self.train_split == 0:
                self.val_idces.append(i)
            else:
                self.train_idces.append(i)
        print('Processed {:d} images'.format(len(self.images)))
        print('%d training points\n%d validation points'%(len(self.train_idces), len(self.val_idces)))
        selected_idces = self.train_idces if self.train else self.val_idces
        self.images = [self.images[i] for i in selected_idces]
        self.poses = self.poses[selected_idces]
    
        for i in range(len(self.images)):
            self.images[i].pose = self.poses[i]
        self.gt_idx = np.stack(selected_idces)

       
        
    def __getitem__(self, index):
        img = None
        while img is None:
            img = load_image(self.images[index].path)
            pose = self.images[index].pose
            index += 1
        index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose) 
        
        draw = np.random.randint(low=1,high=101,size=1)
        if draw > self.real_prob and self.train:
            num_styles = len(self.available_styles)
            style_idx = np.random.choice(num_styles,1)
            style_path = os.path.join(self.style_dir,self.available_styles[style_idx[0]])
            style = load_image(style_path)
        
            ## stylization
            t_list = [t for t in self.transform.__dict__['transforms'] if isinstance(t,transforms.Resize) \
                                                                        or isinstance(t,transforms.CenterCrop) \
                                                                        or isinstance(t,transforms.ToTensor) \
                                                                        or isinstance(t,transforms.Normalize)]

            #img_t = self.transform(img)
            img_t = self.transform(img)
            style_t = style
            content_t = img
            for t in t_list:
                if isinstance(t,transforms.ToTensor):
                    if style_t.size[0] != img_t.shape[-1]:
                        # in case CenterCrop is not contained in self.transform
                        CenterCrop = transforms.CenterCrop(img_t.shape[-2:])
                        style_t = CenterCrop(style_t)
                        content_t = CenterCrop(content_t)
                style_t = t(style_t)
                content_t = t(content_t)
            content_style = torch.stack([content_t,style_t],dim=0)
            return (img_t,content_style,torch.ones(1)),pose
        else:
            img_t = self.transform(img)
            style_t = img_t
            content_t = img_t
            content_style = torch.stack([content_t,style_t],dim=0)
            return (img_t,content_style,torch.zeros(1)),pose


    def __len__(self):
      return self.poses.shape[0]

def main():
    """
    visualizes the dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('../AdaIN/models/decoder.pth'))
    vgg.load_state_dict(torch.load('../AdaIN/models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)

    decoder.to(device)

    num_workers = 2
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])
    inv_normalize = transforms.Normalize(
     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
     )

    data_path = '../data/deepslam_data/AachenDayNight'
    train = False
    dset = AachenDayNight(data_path, train,transform=transform,real_prob=50,style_dir='../data/style_selected')
    print('Loaded AachenDayNight training data, length = {:d}'.format(
    len(dset)))
    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)
    batch_count = 0
    N_batches = 2
    for batch in data_loader:
        real = batch[0][0]
        content_style = batch[0][1]
        content = content_style[:,0]
        style = content_style[:,1]
        style_indc = batch[0][2].squeeze(1)
        if sum(style_indc == 1) > 0:
            with torch.no_grad():
                alpha = 0.5
                assert (0.0 <= alpha <= 1.0)
                content_f = vgg(content[style_indc == 1].cuda())
                style_f = vgg(style[style_indc == 1].cuda())
                feat = adaptive_instance_normalization(content_f, style_f)
                feat = feat * alpha + content_f * (1 - alpha)
                stylized = decoder(feat)
                real[style_indc == 1] = stylized.cpu()
            

        show_batch(make_grid(real, nrow=2, padding=5, normalize=True))
        
        pose = batch[1]
        
        #show_batch(make_grid(style, nrow=1, padding=5, normalize=True))
        batch_count += 1
        if batch_count >= N_batches:
            break

if __name__ == '__main__':
    main()
