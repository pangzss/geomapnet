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
import tqdm


class Sample:
    def __init__(self, path, pose):
        self.path = path
        self.pose = pose


class Triplet:
    def __init__(self, path, pose):
        self.anchor_path = path
        self.anchor_pose = pose
        
        self.pos_paths = []
        self.pos_poses = []

        self.neg_paths = []
        self.neg_poses = []

class AachenTriplet(data_.Dataset):
    def __init__(self, data_path, train,transform=None, 
                target_transform=None, style_dir = None,real_prob = 100,
                train_split=20,seed=0,scene=None):
        np.random.seed(seed)
        self.data_path = data_path
        # MostSimPairs lists 20 other images that share the most number of 
        # points with the indexed image.
        self.MostSimPairs_path = '../data/triplet/Aachen/MostSimPairs.txt'
        # AllPairs lists all the images that share points with the indexed image
        self.AllPairs_path = '../data/triplet/Aachen/AllPairs.txt'
        self.train = train
        #
        self.real_prob = real_prob if self.train else 100
        self.style_dir = style_dir+'_stats_AachenDayNight' if self.train else None
        self.available_styles = os.listdir(self.style_dir) if self.style_dir is not None else None
        print('real_prob: {}.\nstyle_dir: {}\nnum_styles: {}'.format(self.real_prob,self.style_dir,len(self.available_styles) \
                                                                                                if self.style_dir is not None else 0))
        #
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
        
        if not os.path.exists('../data/triplet/Aachen/MostSimPairs_dict.txt'):
            MostSimPairs_file = open(self.MostSimPairs_path)
            lines = MostSimPairs_file.readlines()
            MostSimPairs = [line.split(' ') for line in lines]
            MostSimPairs_dict = {}
            for pair in MostSimPairs:
                if pair[0] not in list(MostSimPairs_dict.keys()):
                    MostSimPairs_dict[pair[0]] = []
                MostSimPairs_dict[pair[0]].append(pair[1].strip())
            with open('../data/triplet/Aachen/MostSimPairs_dict.txt', 'w') as f:
                print(MostSimPairs_dict, file=f)
        else:
            with open('../data/triplet/Aachen/MostSimPairs_dict.txt', 'r') as f:
                MostSimPairs_dict = eval(f.read())
        
        if not os.path.exists('../data/triplet/Aachen/AllPairs_dict.txt'):
            AllPairs_file = open(self.AllPairs_path)
            lines = AllPairs_file.readlines()
            AllPairs = [line.split(' ') for line in lines]
            AllPairs_dict = {}
            for pair in AllPairs:
                if pair[0] not in list(AllPairs_dict.keys()):
                    AllPairs_dict[pair[0]] = []
                AllPairs_dict[pair[0]].append(pair[1].strip())
            with open('../data/triplet/Aachen/AllPairs_dict.txt', 'w') as f:
                print(AllPairs_dict, file=f)
        else:
            with open('../data/triplet/Aachen/AllPairs_dict.txt', 'r') as f:
                AllPairs_dict = eval(f.read())

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
            #path = os.path.join(self.data_path, path[0],'real',path[1])
            path = os.path.join(self.data_path,pose_i[0])
    
            self.images.append(Sample(path,pose))
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
        imgs_poses = {}
        for i in range(len(self.images)):
            self.images[i].pose = self.poses[i]
            path_parts = self.images[i].path.split('/')
            img_name = os.path.join(path_parts[-2],path_parts[-1])
            imgs_poses[img_name] = self.poses[i]

        self.gt_idx = np.stack(selected_idces)
  
        self.triplets = []
    
        # kept for training or val
        kept_images = list(imgs_poses.keys())

        self.triplets_idx = np.zeros(len(kept_images))
        for i,img in enumerate(self.images):

            path_parts = img.path.split('/')
            anchor_name = os.path.join(path_parts[-2],path_parts[-1])
            
            triplet = Triplet(img.path,img.pose)
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
                if pos_names and neg_names:
                    triplet.pos_paths = [os.path.join(self.data_path,pos) for pos in pos_names]
                    triplet.pos_poses = [imgs_poses[pos] for pos in pos_names]

                    triplet.neg_paths = [os.path.join(self.data_path, neg) for neg in neg_names]
                    triplet.neg_poses = [imgs_poses[neg] for neg in neg_names]
                    self.triplets_idx[i] = 1
                else:
                    self.triplets_idx[i] = 0
            #else:
            #    print(anchor_name)

            if self.triplets_idx[i] == 0:
                rand_idx = np.random.randint(0,len(kept_images),2)
                rand_img1 = kept_images[rand_idx[0]]
                rand_img2 = kept_images[rand_idx[1]]
            
                while rand_img1 == anchor_name or rand_img2 == anchor_name:
                    rand_idx = np.random.randint(0,len(kept_images),2)
                    rand_img1 = kept_images[rand_idx[0]]
                    rand_img2 = kept_images[rand_idx[1]]
                triplet.pos_paths = [os.path.join(self.data_path,rand_img1)]
                triplet.pos_poses = [imgs_poses[rand_img1]]
                triplet.neg_paths = [os.path.join(self.data_path,rand_img2)]
                triplet.neg_poses = [imgs_poses[rand_img2]]
                

            self.triplets.append(triplet)
        print('loaded {} triplets for {} data points'.format(len(self.triplets),len(self.images)))

    def get_style(self,img_shape):
        num_styles = len(self.available_styles)
        style_idx = np.random.choice(num_styles,1)
        style_stats_path = os.path.join(self.style_dir,self.available_styles[style_idx[0]])
        style_stats = np.loadtxt(style_stats_path)
        
        style_stats = torch.tensor(style_stats,dtype=torch.float) # 2*512

        return style_stats

    def __getitem__(self, index):
        
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
        #triplet_idx = self.triplets_idx[index]
        #print(anchor.shape,pos.shape,neg.shape)
        real_triplet = torch.stack((pos,anchor,neg),dim=0)
        style_triplet = torch.stack((pos_style,anchor_style,neg_style),dim=0)
        pose_triplet = torch.stack((pos_pose,anchor_pose,neg_pose),dim=0)
        return (real_triplet, style_triplet,style_idx),pose_triplet
        '''
        else:
            img = load_image(self.images[index].path)
            if self.target_transform is not None:
                pose = self.target_transform(self.images[index].pose)
            
            img = self.transform(img)

            return (img,0),pose
        '''
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
    train = False
    dset = AachenTriplet(data_path, train,transform=transform,target_transform=target_transform,
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
        print(data[2])
        print(data[1][:,:,:,0])
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
