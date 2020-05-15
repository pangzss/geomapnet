from __future__ import division
import sys
from torch.utils import data as data_
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

class AachenDayNight(data_.Dataset):
    def __init__(self, data_path, train,skip_images=False,real=False,transform=None, 
                target_transform=None, style_dir = None,real_prob = 100,
                train_split=20,seed=7,scene=None):
        np.random.seed(seed)
        self.data_path = data_path
        self.train = train
        #
        self.real_prob = real_prob if self.train else 100
    
        if self.real_prob != 100:
            self.style_dist = np.loadtxt(os.path.join('..','data',style_dir)) 
            self.mean = torch.tensor(self.style_dist[0],dtype=torch.float)
            self.cov = torch.tensor(self.style_dist[1:],dtype=torch.float)
            u, s, vh = np.linalg.svd(self.cov)
            self.A = np.matmul(u,np.diag(s**0.5))
            self.A = torch.tensor(self.A).float()
        #self.style_dir = style_dir+'_stats_AachenDayNight' if self.train else None
        #self.available_styles = os.listdir(self.style_dir) if self.style_dir is not None else None
        #print('real_prob: {}.\nstyle_dir: {}\nnum_styles: {}'.format(self.real_prob,self.style_dir,len(self.available_styles) \
        #                                                                                        if self.style_dir is not None else 0))
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
            if pose_i[0] == 'db/2048.jpg':
                continue
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
            #num_styles = len(self.available_styles)
            #style_idx = np.random.choice(num_styles,1)
            #style_stats_path = os.path.join(self.style_dir,self.available_styles[style_idx[0]])
            #style_stats = np.loadtxt(style_stats_path)
            
            #style_stats = torch.tensor(style_stats,dtype=torch.float) # 2*512
            embedding = torch.randn(1,1024)
            embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean
            #embedding = np.random.multivariate_normal(self.mean, self.cov,1)
            style_stats = embedding.reshape((2,512))
          
            #img_t = self.transform(img)
            img_t = self.transform(img)
           
            return (img_t,style_stats,torch.ones(1)),pose
        else:
            img_t = self.transform(img)
            style_stats = torch.zeros((2,512))
        
            return (img_t,style_stats,torch.zeros(1)),pose


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

    num_workers = 0
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])
    inv_normalize = transforms.Normalize(
     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
     )
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    data_path = '../data/deepslam_data/AachenDayNight'
    train = True
    dset = AachenDayNight(data_path, train,transform=transform,target_transform=target_transform,
    real_prob=75,style_dir='../data/style_selected')
    print('Loaded AachenDayNight training data, length = {:d}'.format(
    len(dset)))
    data_loader = data_.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)
    batch_count = 0
    N_batches = 20
    for data,poses in data_loader:
        data_shape = data[0].shape
        to_shape = (-1,data_shape[-3],data_shape[-2],data_shape[-1])
        real = data[0].reshape(to_shape)
        #triplet_idx = data[1]
        style_stats = data[1].reshape(-1,2,512)
        style_indc = data[2].view(-1)

      
   
        if sum(style_indc == 1) > 0:
            with torch.no_grad():
                alpha = 1.0
                assert (0.0 <= alpha <= 1.0)
                content_f = vgg(real[style_indc == 1].cuda())
                style_f_stats = style_stats[style_indc == 1].unsqueeze(-1).unsqueeze(-1).cuda()
                #style_f = vgg(style[style_indc == 1].cuda())
                feat = adaptive_instance_normalization(content_f, style_f_stats,style_stats=True)
                feat = feat * alpha + content_f * (1 - alpha)
                stylized = decoder(feat)
                real[style_indc == 1] = stylized.cpu()
            

        show_batch(make_grid(real, nrow=6, padding=5, normalize=True))
        
        
        #show_batch(make_grid(style, nrow=1, padding=5, normalize=True))
        batch_count += 1
        if batch_count >= N_batches:
            break

if __name__ == '__main__':
    main()
