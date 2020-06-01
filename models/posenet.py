"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
from torchvision.utils import save_image
import os
os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

import sys
sys.path.insert(0, '../')

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

class PoseNet(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False):
    super(PoseNet, self).__init__()
    self.droprate = droprate

    # replace the last FC layer in feature extractor
    self.feature_extractor = feature_extractor
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    fe_out_planes = self.feature_extractor.fc.in_features
    self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 3)
    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    
    x = self.feature_extractor(x)
    x = F.relu(x)
    
    if self.droprate > 0:
      x = F.dropout(x,p=self.droprate,training=self.training)
  
    xyz  = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1)

class MapNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, mapnet):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(MapNet, self).__init__()
    self.mapnet = mapnet

  def forward(self, x):
    """
    :param x: image blob (N x T x C x H x W)
    :return: pose outputs
     (N x T x 6)
    """
   
    s = x.size()
    if len(s) == 5:
      x = x.view(-1, *s[2:])
    else:
      x = x.view(-1, *s[1:])
    poses = self.mapnet(x)
    poses = poses.view(s[0], s[1], -1)
   
    return poses



class TriNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, mapnet,layer=4,block=2):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(TriNet, self).__init__()
    self.mapnet = mapnet
    self.feats = None
    self.selected_layer = layer
    self.selected_block = block
    self.hook_layer_forward()
  def hook_layer_forward(self):
        def hook_function(module, input, output):
            self.feats = output
        # Hook the selected layer
        self.mapnet._modules['feature_extractor']._modules['layer'+str(self.selected_layer)][self.selected_block]._modules['conv2'].register_forward_hook(hook_function)

  def forward(self, x):
    """
    :param x: image blob (N x T x C x H x W)
    :return: pose outputs
     (N x T x 6)
    """
   
    s = x.size()
    if len(s) == 5:
      x = x.view(-1, *s[2:])
      poses = self.mapnet(x)
      poses = poses.view(s[0], s[1], -1)
    else:
      x = x.view(-1, *s[1:])
      poses = self.mapnet(x)
    if len(s) == 5:
      self.feats = self.feats.view(s[0],s[1],self.feats.shape[-3],self.feats.shape[-2],self.feats.shape[-1])
    else:
      self.feats = None
    return (poses,self.feats)


class StripNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, stripnet):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(StripNet, self).__init__()
    self.stripnet = stripnet
    self.feats = None
    self.selected_layer = [(1,0),(2,0),(3,0),(3,4),(4,2)]
    for lb in self.selected_layer:
      self.hook_layer_forward(lb[0],lb[1])

  def calc_mean_std(self,feat, eps=1e-5):
      # eps is a small value added to the variance to avoid divide-by-zero.
      size = feat.size()
      assert (len(size) == 4)
      N, C = size[:2]
      feat_var = feat.view(N, C, -1).var(dim=2) + eps
      feat_std = feat_var.sqrt().view(N, C, 1, 1)
      feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
      return feat_mean, feat_std
  def conditional_instance_norm(self,feats_content,feats_style):
      size = feats_content.size()

      style_mean,style_std = self.calc_mean_std(feats_style)
      content_mean,content_std = self.calc_mean_std(feats_content)
      normalized_feat = (feats_content - content_mean.expand(
      size)) / content_std.expand(size)
      stylized = normalized_feat * style_std.expand(size) + style_mean.expand(size)
      stylized = 0.5*feats_content + 0.5*stylized
      return stylized 
  def hook_layer_forward(self,layer,block):
        def hook_function(module, input, output):
            if output.shape[0] != 1:
              if layer == 1 and block == 0:
                s = output.size()
                output = output.reshape(-1,4,s[-3],s[-2],s[-1])
                feats_triplet = output[:,:3].clone()
                feats_content = output[:,1].clone()
                feats_style = output[:,3].clone()
              
                stylized = self.conditional_instance_norm(feats_content,feats_style)

                output = torch.cat([feats_triplet,feats_style[:,None,...],stylized[:,None,...]],dim=1)
                s = output.size()
                output = output.view(-1,*s[2:])
                
              elif layer == 3 and block == 4:
                s = output.size()
                output = output.reshape(-1,5,s[-3],s[-2],s[-1])
                feats_triplet = output[:,:3].clone()
                feats_content = output[:,-1].clone()
                feats_style = output[:,3].clone()

                stylized = self.conditional_instance_norm(feats_content,feats_style)

                output = torch.cat([feats_triplet,stylized[:,None,...]],dim=1)
                s = output.size()
                output = output.view(-1,*s[2:])
                
              elif layer == 4 and block == 2:
                self.feats = output
  
              else:
                s = output.size()
                
                output = output.reshape(-1,5,s[-3],s[-2],s[-1])
                feats_triplet = output[:,:3].clone()
                feats_content = output[:,-1].clone()
                feats_style = output[:,3].clone()

                stylized = self.conditional_instance_norm(feats_content,feats_style)

                output = torch.cat([feats_triplet,feats_style[:,None,...],stylized[:,None,...]],dim=1)
                s = output.size()
          
                output = output.view(-1,*s[2:])

    
              return output
             
            
            else:
              pass

        # Hook the selected layer
        if layer == 4 and block == 2:
          self.stripnet._modules['feature_extractor']._modules['layer'+str(layer)][block]._modules['conv2'].register_forward_hook(hook_function)
        else:
          self.stripnet._modules['feature_extractor']._modules['layer'+str(layer)][block].register_forward_hook(hook_function)

  def forward(self, x):
    """
    :param x: image blob (N x T x C x H x W)
    :return: pose outputs
     (N x T x 6)
    """
   
    s = x.size()
    if len(s) == 5:
      x = x.view(-1, *s[2:])
      poses = self.stripnet(x)
      poses = poses.view(s[0], s[1], -1)
    else:
      x = x.view(-1, *s[1:])
      poses = self.stripnet(x)
    if len(s) == 5:
      self.feats = self.feats.view(s[0],s[1],self.feats.shape[-3],self.feats.shape[-2],self.feats.shape[-1])
    else:
      self.feats = None
    return (poses,self.feats)


class SLocNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """

  def __init__(self, mapnet, decoder):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(SLocNet, self).__init__()
    self.mapnet = mapnet
    self.decoder = decoder
    self.feats = None
    self.stylized = None
    self.style_feats = {}
    self.mode = 'feature_extraction' # 'loss_induction'
    self.style_layers = {(1, 0), (2, 0), (2, 3),(3,3)}  #, (4, 2)]
    self.content_layers = {(3, 3), (4, 2)}
    for lb in self.style_layers.union(self.content_layers):
      self.hook_layer_forward(lb[0], lb[1])
      self.style_feats[lb[0]] = {}

    self.content_loss = 0
    self.style_loss_1 = None
    self.style_loss_2 = None
    self.style_loss_3 = None
    self.style_loss_4 = None
    self.mse = nn.MSELoss()
    self.counter = 0
    self.notch = 0

  def calc_mean_std(self, feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

  def conditional_instance_norm(self, feats_content, feats_style):
    size = feats_content.size()

    style_mean, style_std = self.calc_mean_std(feats_style)
    content_mean, content_std = self.calc_mean_std(feats_content)
    normalized_feat = (feats_content - content_mean.expand(
      size)) / content_std.expand(size)
    stylized = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    #stylized = 0.5 * feats_content + 0.5 * stylized
    return stylized

  def hook_layer_forward(self, layer, block):
    def hook_function(module, input, output):
      if output.shape[0] != 1:

        if self.mode == 'feature_extraction':

          if layer == 3 and block == 3:
            s = output.size()
            output = output.reshape(-1, 4, s[-3], s[-2], s[-1])
            self.style_feats[layer][block] = output[:,3].clone()
            content_feats = output[:,1].clone()
            self.stylized = self.conditional_instance_norm(content_feats,self.style_feats[layer][block])

            output = output[:, :3]
            output = output.reshape(-1, *s[1:])

          elif layer == 4 and block == 2:
            self.feats = output
          else:
            s = output.size()
            output = output.reshape(-1, 4, s[-3], s[-2], s[-1])
            self.style_feats[layer][block] = output[:, 3].clone()
            output = output.reshape(-1, *s[1:])

        elif self.mode == 'loss_induction':
           def GramMatrix(input_):
              b, c, h, w = input_.size()
              stacked = input_.view(b, c, h * w)
              G = torch.bmm(stacked, stacked.transpose(1, 2))
              G.div_(h * w)
              return G
           if layer == 4 and block == 2:
             s = self.feats.size()
             self.feats = self.feats.reshape(-1,3,s[-3],s[-2],s[-1])
             self.feats = torch.cat([self.feats,output[:,None]],dim=1)

           elif layer == 3 and block == 3:
             self.style_loss_4 = self.mse(GramMatrix(self.style_feats[layer][block]), GramMatrix(output))
             self.content_loss = self.mse(output, self.stylized)
           else:

             if layer == 1:
               #s_mean, s_std = self.calc_mean_std(self.style_feats[layer])
               #t_mean, t_std = self.calc_mean_std(output)
               #s_embedding = torch.cat([s_mean, s_std], dim=1)
               #t_embedding = torch.cat([t_mean, t_std], dim=1)
               self.style_loss_1 = self.mse(GramMatrix(self.style_feats[layer][block]), GramMatrix(output))
               #self.style_loss_1 = self.mse(GramMatrix(self.style_feats[layer]), GramMatrix(output))
             elif layer == 2 and block == 0:
               self.style_loss_2 = self.mse(GramMatrix(self.style_feats[layer][block]), GramMatrix(output))
             else:
               self.style_loss_3 = self.mse(GramMatrix(self.style_feats[layer][block]), GramMatrix(output))
        else:
          NotImplementedError

        return output
      else:
        pass

    # Hook the selected layer

    self.mapnet._modules['feature_extractor']._modules['layer' + str(layer)][block].register_forward_hook(
      hook_function)

  def forward(self, x):
    """
    :param x: image blob (N x T x C x H x W)
    :return: pose outputs
     (N x T x 6)
    """

    s = x.size()
    if len(s) == 5:
      self.counter += 1
      x = x.view(-1, *s[2:])
      self.mode = 'feature_extraction'
      poses = self.mapnet(x)
      poses = poses.view(s[0], 3, -1)

      self.mode = 'loss_induction'
      t = self.decoder(self.stylized,3,0)
      if self.counter == 100:
        self.notch += 1
        self.counter = 0
        to_save = t[0].clone().cpu().detach()
        save_image(to_save,'../scripts/logs/generated_imgs/{}.jpg'.format(self.notch),normalize=True)

      pose_t = self.mapnet(t).view(s[0],1,-1)

      poses = torch.cat([poses,pose_t],dim=1)

      return poses, [self.feats,self.content_loss, self.style_loss_1+self.style_loss_2+self.style_loss_3]
    else:
      x = x.view(-1, *s[1:])
      poses = self.mapnet(x)


      return poses,[None,0,0]

if __name__ == '__main__':
  from torchvision import models
  from torchvision import transforms
  from dataset_loaders.utils import load_image
  from resizeConv_resnet import resizeConv_resnet
  feature_extractor = models.resnet34(pretrained=True)
  posenet = PoseNet(feature_extractor, droprate=0, pretrained=True)
  decoder = resizeConv_resnet()
  model = SLocNet(mapnet=posenet,decoder=decoder)

  num_workers = 0
  transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),

  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])

  data_path = '../data/deepslam_data/AachenDayNight/db'
  img_dir = os.listdir(data_path)[0]
  img = load_image(os.path.join(data_path,img_dir))
  img = transform(img)

  img = torch.ones(10,4,3,256,256)
  pose,feats = model(img)
  print(pose.shape)


