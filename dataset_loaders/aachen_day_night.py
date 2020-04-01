from __future__ import division
import sys
from torch.utils import data
import numpy as np
import ntpath
import random
import torch
import transforms3d.quaternions as txq
import os
import tqdm
from PIL import Image

sys.path.insert(0, '../')
from common.pose_utils import process_poses_quaternion
from common.utils import load_image

available_styles = [
    'goeritz',
    'asheville',
    'mondrian',
    'scene_de_rue',
    'flower_of_life',
    'antimonocromatismo',
    'woman_with_hat_matisse',
    'trial',
    'sketch',
    'picasso_self_portrait',
    'picasso_seated_nude_hr',
    'la_muse',
    'contrast_of_forms',
    'brushstrokes',
    'the_resevoir_at_poitiers',
    'woman_in_peasant_dress'
]

class Sample:
    def __init__(self, path, pose):
        self.path = path
        self.pose = pose
        self.stylized = None

class AachenDayNight(data.Dataset):
    def __init__(self, data_path, train,skip_images=False,real=False,transform=None, 
                target_transform=None, num_styles=0,
                train_split=20,seed=7,scene=None):
        np.random.seed(seed)
        self.data_path = data_path
        self.train = train
        self.num_styles = num_styles
        self.skip_images = skip_images
        self.train_split = train_split
        print('-train status: {}. \n-train&val ratio: {}'.\
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
            path = pose_i[0].split('/')
            path = os.path.join(self.data_path, path[0],'real',path[1])
            
            self.images.append(Sample(path,pose))
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
            if self.num_styles != 0:
                which_style = np.random.randint(low=0,high=self.num_styles,size=1)
                styl_img = load_image(self.images[index].stylized[which_style[0]])
            else:
                styl_img = None
            pose = self.images[index].pose
            index += 1
        index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose) 
        
        # 1x(num_styles+1)xCxHxW
        img = self.transform(img)
        styl_img = img.clone()
            #styl_imgs = torch.stack(styl_imgs,dim=0)
            #out = torch.cat([img,styl_imgs],dim=0)
     
        #styl_imgs = [(i,index) for i in range(4)] 
        return (img,styl_img),pose

    def __len__(self):
      return self.poses.shape[0]

def main():
    """
    visualizes the dataset
    """
    from common.vis_utils import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms

    num_workers = 6
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])
    
    data_path = '../data/deepslam_data/AachenDayNight'
    train = True
    dset = AachenDayNight(data_path, train,transform=transform, num_styles=16)
    print('Loaded AachenDayNight training data, length = {:d}'.format(
    len(dset)))
    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)
    batch_count = 0
    N_batches = 2
    for batch in data_loader:
    
        real = batch[0][0]
        style = batch[0][1]
        
        updated_batch = torch.zeros_like(real)
        real_prob = 80
    
        N = real.shape[0]
        draw = np.random.randint(low=1,high=101,size=N)
        style_idces = draw > real_prob
        real_idces = draw <= real_prob

        updated_batch[style_idces] = style[style_idces]
        updated_batch[real_idces] = real[real_idces]

        pose = batch[1]
        print('Minibatch {:d}'.format(batch_count))
        show_batch(make_grid(torch.cat([updated_batch,style],dim=0), nrow=2, padding=5, normalize=True))
        #show_batch(make_grid(style, nrow=1, padding=5, normalize=True))
        batch_count += 1
        if batch_count >= N_batches:
            break

if __name__ == '__main__':
    main()
