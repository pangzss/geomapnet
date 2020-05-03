from __future__ import division
import sys
sys.path.insert(0, '../')
from torch.utils import data as data_
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

class Triplet:
    def __init__(self, path, pose):
        self.anchor_path = path
        self.anchor_pose = pose
        
        self.pos_paths = []
        self.pos_poses = []

        self.neg_paths = []
        self.neg_poses = []

class CambridgeTriplet(data_.Dataset):
    
    def __init__(self, data_path, train, overfit=None, scene='ShopFacade',
                seed=7, real=False,transform=None, target_transform=None,
                style_dir = None,real_prob = 100, min_perceptual=False):
 
        np.random.seed(seed)
        self.data_path = data_path
        self.scene = scene
        # MostSimPairs lists 20 other images that share the most number of 
        # points with the indexed image.
        self.triplet_path = os.path.join('..','data','triplet',self.scene)
        self.MostSimPairs_path = os.path.join(self.triplet_path,'MostSimPairs.txt')
        # AllPairs lists all the images that share points with the indexed image
        self.AllPairs_path = os.path.join(self.triplet_path,'AllPairs.txt')

        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        #
        self.real_prob = real_prob if self.train else 100
        self.style_dist = np.loadtxt(os.path.join('..','data',style_dir)) if self.train else None
        self.mean = torch.tensor(self.style_dist[0],dtype=torch.float) if self.train else None
        self.cov = torch.tensor(self.style_dist[1:],dtype=torch.float) if self.train else None 
        u, s, vh = np.linalg.svd(self.cov)
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float()
        #
        
        self.min_perceptual = min_perceptual

        # load triplets
        if not os.path.exists(os.path.join(self.triplet_path, 'MostSimPairs_dict.txt')):
            MostSimPairs_file = open(self.MostSimPairs_path)
            lines = MostSimPairs_file.readlines()
            MostSimPairs = [line.split(' ') for line in lines]
            MostSimPairs_dict = {}
            for pair in MostSimPairs:
                if pair[0] not in list(MostSimPairs_dict.keys()):
                    MostSimPairs_dict[pair[0]] = []
                MostSimPairs_dict[pair[0]].append(pair[1].strip())
            with open(os.path.join(self.triplet_path, 'MostSimPairs_dict.txt'), 'w') as f:
                print(MostSimPairs_dict, file=f)
        else:
            with open(os.path.join(self.triplet_path, 'MostSimPairs_dict.txt'), 'r') as f:
                MostSimPairs_dict = eval(f.read())
        
        if not os.path.exists(os.path.join(self.triplet_path,'AllPairs_dict.txt')):
            AllPairs_file = open(self.AllPairs_path)
            lines = AllPairs_file.readlines()
            AllPairs = [line.split(' ') for line in lines]
            AllPairs_dict = {}
            for pair in AllPairs:
                if pair[0] not in list(AllPairs_dict.keys()):
                    AllPairs_dict[pair[0]] = []
                AllPairs_dict[pair[0]].append(pair[1].strip())
            with open(os.path.join(self.triplet_path,'AllPairs_dict.txt'), 'w') as f:
                print(AllPairs_dict, file=f)
        else:
            with open(os.path.join(self.triplet_path,'AllPairs_dict.txt'), 'r') as f:
                AllPairs_dict = eval(f.read())

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
        
        imgs_poses = {}
        for p in self.points:
            pose = process_poses_quaternion(np.asarray([p.position]), np.asarray([p.rotation]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose.flatten())
            path_parts = p.img_path.split('/')
            img_name = os.path.join(path_parts[-2],path_parts[-1])
            imgs_poses[img_name] = p.pose

        if self.train:
            self.triplets = []
        
            # kept for training or val
            kept_images = list(imgs_poses.keys())

            for i,img in enumerate(self.points):

                path_parts = img.img_path.split('/')
                anchor_name = os.path.join(path_parts[-2],path_parts[-1])
                
                triplet = Triplet(os.path.join(self.data_path,self.scene,img.img_path),img.pose)
                # only keep those pos in training or val set

                if anchor_name in list(MostSimPairs_dict.keys()) and \
                    anchor_name in list(AllPairs_dict.keys()):
                    pos_names = set(MostSimPairs_dict[anchor_name]).intersection(set(kept_images)) - set([anchor_name])

                    #print('pos:',pos_names)
                    # only keep those neg in training or val set
                    all_pos_names = set(AllPairs_dict[anchor_name]).intersection(set(kept_images))
                    # no shared points and in training or val set
                    neg_names = set(kept_images) - all_pos_names -set([anchor_name])

                    #print('neg:',neg_names)

                    triplet.pos_paths = [os.path.join(self.data_path,self.scene,pos) for pos in pos_names]
                    triplet.pos_poses = [imgs_poses[pos] for pos in pos_names]

                    triplet.neg_paths = [os.path.join(self.data_path,self.scene,neg) for neg in neg_names]
                    triplet.neg_poses = [imgs_poses[neg] for neg in neg_names]

                    if len(triplet.pos_paths) == 0 or len(triplet.neg_paths) == 0:
                        continue
                self.triplets.append(triplet)
            print('loaded {} triplets for {} data points'.format(len(self.triplets),len(self.points)))

    def get_style(self,img_shape):
        embedding = torch.randn(1,1024)
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean
        #embedding = np.random.multivariate_normal(self.mean, self.cov,1)
        style_stats = embedding.reshape((2,512))

        return style_stats

    def __getitem__(self, index):
        if self.train:
            triplet = self.triplets[index]
            #print('anchor: ',triplet.anchor_path)
            anchor = load_image(triplet.anchor_path)
            anchor_pose = triplet.anchor_pose
           

            pos_idx = np.random.randint(low=0,high=len(triplet.pos_paths),size=1).item()
            pos = load_image(triplet.pos_paths[pos_idx])
            #print('pos:', triplet.pos_paths[pos_idx])
            pos_pose = triplet.pos_poses[pos_idx]

            neg_idx = np.random.randint(low=0,high=len(triplet.neg_paths),size=1).item()
            neg = load_image(triplet.neg_paths[neg_idx])
            neg_pose = triplet.neg_poses[neg_idx]
            #print('neg:',triplet.neg_paths[neg_idx])
            if self.target_transform is not None:
                anchor_pose = self.target_transform(anchor_pose)
                pos_pose = self.target_transform(pos_pose)
                neg_pose = self.target_transform(neg_pose)
            
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)
            
            if not self.min_perceptual:
                style_idx = torch.zeros(3)
                draw = np.random.randint(low=1,high=101,size=1)
                if draw > self.real_prob and self.train:
                    anchor_style = self.get_style(anchor.shape)
                    style_idx[1] = 1
                else:
                    anchor_style = torch.zeros(2,512)
                
                draw = np.random.randint(low=1,high=101,size=1)
                if draw > self.real_prob and self.train:
                    pos_style = self.get_style(pos.shape)
                    style_idx[0] = 1
                else:
                    pos_style = torch.zeros(2,512)

                draw = np.random.randint(low=1,high=101,size=1)
                if draw > self.real_prob and self.train:
                    neg_style = self.get_style(neg.shape)
                    style_idx[2] = 1
                else:
                    neg_style = torch.zeros(2,512)
            else:
                style_idx = torch.zeros(3)
                
                anchor_style = self.get_style(anchor.shape)
                style_idx[1] = 1
            
                pos_style = torch.zeros(2,512)
                neg_style = torch.zeros(2,512)
            #triplet_idx = self.triplets_idx[index]
            #print(anchor.shape,pos.shape,neg.shape)
            real_triplet = torch.stack((pos,anchor,neg),dim=0)
            style_triplet = torch.stack((pos_style,anchor_style,neg_style),dim=0)
            pose_triplet = torch.stack((pos_pose,anchor_pose,neg_pose),dim=0)
            return (real_triplet, style_triplet,style_idx),pose_triplet
        else:
           
            point = self.points[index]
            img = load_image(os.path.join(self.data_path,self.scene,point.img_path))
            pose = point.pose

            if self.target_transform is not None:
                pose = self.target_transform(pose) 
            img_t = self.transform(img)

            return (img_t, 0,0),pose
    
    def __len__(self):
        return len(self.triplets) if self.train else len(self.points)
    
    
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
    data_path = '../data/deepslam_data/Cambridge'
    scene = 'ShopFacade'
    train = True
    min_perceptual = True
    dset = CambridgeTriplet(data_path, train,scene=scene,transform=transform, target_transform=target_transform,
    real_prob=100,style_dir='../data/pbn_train_embedding_dist.txt',min_perceptual=min_perceptual)
    print('Loaded Cambridge training data, length = {:d}'.format(
    len(dset)))
    data_loader = data_.DataLoader(dset, batch_size=3, shuffle=True,
    num_workers=num_workers)
    batch_count = 0
    N_batches = 5
    for data,poses in data_loader:
        data_shape = data[0].shape
        to_shape = (-1,data_shape[-3],data_shape[-2],data_shape[-1])
        real = data[0].reshape(to_shape)
        #triplet_idx = data[1]
        style_stats = data[1].reshape(-1,2,512)
        style_indc = data[2].view(-1)
   
        if sum(style_indc == 1) > 0:
            with torch.no_grad():
                alpha = 0.5
                assert (0.0 <= alpha <= 1.0)
                content_f = vgg(real[style_indc == 1].cuda())
                style_f_stats = style_stats[style_indc == 1].unsqueeze(-1).unsqueeze(-1).cuda()
                #style_f = vgg(style[style_indc == 1].cuda())
                feat = adaptive_instance_normalization(content_f, style_f_stats,style_stats=True)
                feat = feat * alpha + content_f * (1 - alpha)
                stylized = decoder(feat).cpu()
                if not min_perceptual:
                      real[style_indc == 1] = stylized

        real = real.reshape(data_shape)
        if min_perceptual:
            stylized = stylized[:,None,...]
            real = torch.cat([real,stylized],dim=1)
    
        show_batch(make_grid(real.reshape(to_shape), nrow=4, padding=5, normalize=True))
        
        
        #show_batch(make_grid(style, nrow=1, padding=5, normalize=True))
        batch_count += 1
        if batch_count >= N_batches:
            break

if __name__ == '__main__':
    main()