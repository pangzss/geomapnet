from __future__ import division
import sys
sys.path.insert(0, '../')
from torch.utils import data
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
from common.pose_utils import process_poses_quaternion
from common.vis_utils import show_batch, show_stereo_batch
import transforms3d.quaternions as txq
import os
from dataset_loaders.utils import load_image
from AdaIN import net
from AdaIN.function import adaptive_instance_normalization
import torch


class camerapoint:
    def __init__(self, position = np.zeros(3), rotation = np.zeros(4), img_path = None):
        self.position = position
        self.rotation = rotation
        self.img_path = img_path
        self.pose = None
        
    def set_pose(self, pose):
        self.pose = pose
        del self.position, self.rotation
        
    def __str__(self):
        return self.img_path+' '+str(self.pose)


class Cambridge(data.Dataset):
    
    def __init__(self, data_path, train, overfit=None, scene='ShopFacade',
                seed=7, real=False,transform=None, target_transform=None,
                style_dir = None,real_prob = 100,mask=False):
 
        np.random.seed(seed)
        self.data_path = data_path
        self.scene = scene
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.mask = mask
        print(self.mask)
        #
        self.real_prob = real_prob
        if self.real_prob != 100:
            self.style_dist = np.loadtxt(os.path.join('..','data',style_dir)) 
            self.mean = torch.tensor(self.style_dist[0],dtype=torch.float32)
            self.cov = self.style_dist[1:]
            u, s, vh = np.linalg.svd(self.cov)
            self.A = np.matmul(u,np.diag(s**0.5))
            self.A = torch.tensor(self.A).float()
        
        #self.style_dir = style_dir+'_stats_'+scene if self.train else None
        #self.available_styles = os.listdir(self.style_dir) if self.style_dir is not None else None
        #print('real_prob: {}.\nstyle_dir: {}\nnum_styles: {}'.format(self.real_prob,self.style_dir,len(self.available_styles) \
        #                                                                                        if self.style_dir is not None else 0))
        #
        
        if self.train:
        
            training_file = open(os.path.join(self.data_path, self.scene, 'dataset_train.txt'), 'r')
        else:
            training_file = open(os.path.join(self.data_path, self.scene, 'dataset_test.txt'), 'r')
            
        lines = training_file.readlines()
        lines = [x.strip() for x in lines]
        
        self.points = []
        for l in lines[3:]:
            ls = l.split(' ')
            pose = [float(x) for x in ls[1:]]
            p = camerapoint(position = pose[:3], rotation = pose[3:], img_path=ls[0])
            self.points.append(p)
        print('Loaded %d points'%len(self.points))
            
        pose_stats_filename = os.path.join('../data/Cambridge', self.scene, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            pos_list = []
            for i in self.points:
                pos_list.append(i.position)
            pos_list = np.asarray(pos_list)
            mean_t = pos_list.mean(axis=0)
            std_t = pos_list.std(axis=0)
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
            #print('Saved pose stats to %s'%pose_stats_filename)
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        for p in self.points:
            pose = process_poses_quaternion(np.asarray([p.position]), np.asarray([p.rotation]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose.flatten())

            
    def __getitem__(self, index):
        img = None
        while img is None:
            point = self.points[index]
            img = load_image(os.path.join(self.data_path,self.scene,point.img_path))
            pose = point.pose
            index += 1
        index -= 1
        
        if self.target_transform is not None:
            pose = self.target_transform(pose) 
        if self.mask:
            path = point.img_path.split('/')
            
            mask = load_image(os.path.join(self.data_path,self.scene,path[0]+'_masks','mask_'+path[1]))
            
            mask = torch.tensor(np.asarray(mask)[:,:,0],dtype=torch.bool)


        draw = np.random.randint(low=1,high=101,size=1)
        if draw > self.real_prob and self.train:
            #num_styles = len(self.available_styles)
            style_stats = np.empty(0)
            while len(style_stats) == 0:
                #style_idx = np.random.choice(num_styles,1)
                #style_stats_path = os.path.join(self.style_dir,self.available_styles[style_idx[0]])
                #style_stats = np.loadtxt(style_stats_path)
                
                #style_stats = torch.tensor(style_stats,dtype=torch.float) # 2*512
                embedding = torch.randn(1,1024)
                embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean
                #embedding = np.random.multivariate_normal(self.mean, self.cov,1)
                style_stats = embedding.view((2,512))
            '''
            ## stylization
            t_list = [t for t in self.transform.__dict__['transforms'] if isinstance(t,transforms.Resize) \
                                                                        or isinstance(t,transforms.CenterCrop) \
                                                                        or isinstance(t,transforms.ToTensor) \
                                                                        or isinstance(t,transforms.Normalize)]
            '''
            img_t = self.transform(img)

            '''
            style_t = style
           
            for t in t_list:
        
                if isinstance(t,transforms.Resize):
                    Resize = transforms.Resize(img_t.shape[-2:])
                    style_t = Resize(style_t)
                    continue
                style_t = t(style_t)
            '''
            if self.mask:
                return (img_t,style_stats,torch.ones(1),mask),pose
            else:
                return (img_t,style_stats,torch.ones(1)),pose
        else:
            
            img_t = self.transform(img)
            style_stats = torch.zeros((2,512))
            
            if self.mask:
                return (img_t,style_stats,torch.zeros(1),mask),pose
            else:
                return (img_t,style_stats,torch.zeros(1)),pose
    
    def __len__(self):
        return len(self.points)
    
    
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
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

    inv_normalize = transforms.Normalize(
     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
     )
    
    data_path = '../data/deepslam_data/Cambridge'
    scene = 'KingsCollege'
    train = True
    dset = Cambridge(data_path, train,scene=scene,transform=transform,real_prob=0,style_dir='pbn_test_embedding_dist.txt')
    print('Loaded Cambridge training data, length = {:d}'.format(
    len(dset)))
    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)
    batch_count = 0
    N_batches = 10
    for batch in data_loader:
        real = batch[0][0]
        print(real.shape)
        style_stats = batch[0][1]
        style_indc = batch[0][2].squeeze(1)

        
        if sum(style_indc == 1) > 0:
            with torch.no_grad():
                alpha = 0.5
                assert (0.0 <= alpha <= 1.0)
                content_f = vgg(real[style_indc == 1].cuda())
                style_f_stats = style_stats[style_indc == 1].unsqueeze(-1).unsqueeze(-1).cuda()
                #style_f = vgg(style[style_indc == 1].cuda())
                feat = adaptive_instance_normalization(content_f, style_f_stats,style_stats=True)
                feat = feat * alpha + content_f * (1 - alpha)
                stylized = decoder(feat)
                real[style_indc == 1] = stylized.cpu()
            

        show_batch(make_grid(real, nrow=5, padding=5, normalize=True))
        
        pose = batch[1]
         
        #show_batch(make_grid(style, nrow=1, padding=5, normalize=True))
        batch_count += 1
        if batch_count >= N_batches:
            break
if __name__ == '__main__':
    main()