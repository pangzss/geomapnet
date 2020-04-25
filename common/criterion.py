"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
This module implements the various loss functions (a.k.a. criterions) used
in the paper
"""

from common import pose_utils
import torch
from torch import nn
import torch.nn.functional as F
class Loss:
  def __init__(self, abs_loss=(torch.tensor(0.0),torch.tensor(0.0)), vo_loss=torch.tensor(0.0), 
              triplet_loss=torch.tensor(0.0),perceptual_loss=torch.tensor(0.0),final_loss=torch.tensor(0.0)):
        self.abs_loss = abs_loss
        self.vo_loss = vo_loss
        self.triplet_loss = triplet_loss
        self.perceptual_loss = perceptual_loss
        self.final_loss = final_loss
class QuaternionLoss(nn.Module):
  """
  Implements distance between quaternions as mentioned in
  D. Huynh. Metrics for 3D rotations: Comparison and analysis
  """
  def __init__(self):
    super(QuaternionLoss, self).__init__()

  def forward(self, q1, q2):
    """
    :param q1: N x 4
    :param q2: N x 4
    :return: 
    """
    loss = 1 - torch.pow(pose_utils.vdot(q1, q2), 2)
    loss = torch.mean(loss)
    return loss

class PoseNetCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
      saq=0.0, learn_beta=False):
    super(PoseNetCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

  def forward(self, pred, targ):
    """
    :param pred: N x 7
    :param targ: N x 7
    :return: 
    """
  
    loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + \
      self.sax +\
     torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) +\
      self.saq
    loss_ = Loss(final_loss=loss)
    return loss_

class MapNetCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False):
    """
    Implements L_D from eq. 2 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    """
    super(MapNetCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

  def forward(self, pred, targ):
    """
    :param pred: N x T x 6
    :param targ: N x T x 6
    :return:
    """
    # absolute pose loss
    s = pred.size()
    t_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],targ.view(-1, *s[2:])[:, :3]) + self.sax
    q_loss = torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],targ.view(-1, *s[2:])[:, 3:]) + self.saq

    abs_loss = t_loss + q_loss
    # get the VOs
    pred_vos = pose_utils.calc_vos_simple(pred)
    targ_vos = pose_utils.calc_vos_simple(targ)

    # VO loss
    s = pred_vos.size()
    vo_loss = \
      torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                            targ_vos.view(-1, *s[2:])[:, :3]) + \
      self.srx + \
      torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                            targ_vos.view(-1, *s[2:])[:, 3:]) + \
      self.srq

    # total loss
    loss_ = Loss(abs_loss=(t_loss,q_loss),vo_loss=vo_loss,final_loss=abs_loss+vo_loss)

    return loss_

class MapNetOnlineCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False,
               gps_mode=False):
    """
    Implements L_D + L_T from eq. 4 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    :param gps_mode: If True, uses simple VO and only calculates VO error in
    position
    """
    super(MapNetOnlineCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)
    self.gps_mode = gps_mode

  def forward(self, pred, targ):
    """
    targ contains N groups of pose targets, making the mini-batch.
    In each group, the first T poses are absolute poses, used for L_D while
    the next T-1 are relative poses, used for L_T
    All the 2T predictions in pred are absolute pose predictions from MapNet,
    but the last T predictions are converted to T-1 relative predictions using
    pose_utils.calc_vos()
    :param pred: N x 2T x 7
    :param targ: N x 2T-1 x 7
    :return:
    """
    s = pred.size()
    T = s[1] / 2
    pred_abs = pred[:, :T, :].contiguous()
    pred_vos = pred[:, T:, :].contiguous()  # these contain abs pose predictions for now
    targ_abs = targ[:, :T, :].contiguous()
    targ_vos = targ[:, T:, :].contiguous()  # contain absolute translations if gps_mode

    # absolute pose loss
    pred_abs = pred_abs.view(-1, *s[2:])
    targ_abs = targ_abs.view(-1, *s[2:])
    abs_loss =\
      torch.exp(-self.sax) * self.t_loss_fn(pred_abs[:, :3], targ_abs[:, :3]) + \
      self.sax + \
      torch.exp(-self.saq) * self.q_loss_fn(pred_abs[:, 3:], targ_abs[:, 3:]) + \
      self.saq

    # get the VOs
    if not self.gps_mode:
      pred_vos = pose_utils.calc_vos(pred_vos)

    # VO loss
    s = pred_vos.size()
    pred_vos = pred_vos.view(-1, *s[2:])
    targ_vos = targ_vos.view(-1, *s[2:])
    idx = 2 if self.gps_mode else 3
    vo_loss = \
      torch.exp(-self.srx) * self.t_loss_fn(pred_vos[:, :idx], targ_vos[:, :idx]) + \
      self.srx
    if not self.gps_mode:
      vo_loss += \
        torch.exp(-self.srq) * self.q_loss_fn(pred_vos[:, 3:], targ_vos[:, 3:]) + \
        self.srq

    # total loss
    loss = abs_loss + vo_loss
    return loss

class TripletCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0,sc=0.0, sp=0.0,
                learn_beta=False, learn_gamma=False, 
                learn_sigma=False):
    """
    Implements L_D from eq. 2 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    """
    super(TripletCriterion, self).__init__()
    from contextual_loss.CX_distance import CX_sim
    self.CS = CX_sim
    self.margin = 0.5
    self.sc = nn.Parameter(torch.Tensor([sc]), requires_grad=learn_sigma)
    self.sp = nn.Parameter(torch.Tensor([sp]), requires_grad=learn_sigma)

    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

  def forward(self, output, targ):
    """
    :param pred: N x T x 6
    :param targ: N x T x 6
    :return:
    """
    pred = output[0]
    feats = output[1]
    s = pred.size()
    if s[1] == 4:
      pred_style = pred[:,3].clone()
      targ_style = targ[:,1].clone()
      pred_tuple = torch.stack([pred[:,1],pred[:,3]],dim=1)
      targ_tuple = torch.stack([targ[:,1],targ[:,1]],dim=1)

      pred = pred[:,:3].clone()
    # contextual triplet loss
    triplet_loss = torch.tensor(0.0)
    if feats is not None:
      sim_ap = self.CS(feats[:,1],feats[:,0])
      sim_an = self.CS(feats[:,1],feats[:,2])
      triplet_loss = torch.mean(F.relu(sim_an-sim_ap+self.margin))
    # perceptual loss
    perceptual_loss = torch.tensor(0.0)
    if feats is not None and s[1]==4:
      real_feats = feats[:,1]
      style_feats = feats[:,3]

      N,C,H,W = real_feats.shape

      real_feats = torch.reshape(real_feats,(N,C,H*W))
      style_feats = torch.reshape(style_feats,(N,C,H*W))

      real_feats_sq = torch.sum(real_feats**2,dim=-1)
      style_feats_sq = torch.sum(style_feats**2,dim=-1)
      product = torch.sum(real_feats*style_feats,dim=-1)

      perceptual_loss = torch.sum((real_feats_sq - 2*product + style_feats_sq),dim=1)/(C*H*W)
      perceptual_loss = torch.mean(perceptual_loss)

    # absolute pose loss
    t_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, s[-1])[:, :3],targ.view(-1, s[-1])[:, :3]) + self.sax
    q_loss = torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, s[-1])[:, 3:],targ.view(-1, s[-1])[:, 3:]) + self.saq
    
    t_loss_style  = torch.tensor(0.0)
    q_loss_style = torch.tensor(0.0)
    if feats is not None and s[1] == 4:
      t_loss_style = torch.exp(-self.sax) * self.t_loss_fn(pred_style.view(-1, s[-1])[:, :3],targ_style.view(-1, s[-1])[:, :3]) + self.sax
      q_loss_style = torch.exp(-self.saq) * self.q_loss_fn(pred_style.view(-1, s[-1])[:, 3:],targ_style.view(-1, s[-1])[:, 3:]) + self.saq

    abs_loss = t_loss + q_loss + t_loss_style + q_loss_style

    # vo loss 
    # get the VOs
    vo_loss = torch.tensor(0.0)
    if feats is not None:
      pred_vos = pose_utils.calc_vos_simple(pred)
      targ_vos = pose_utils.calc_vos_simple(targ)

      pred_vos_tuple = pose_utils.calc_vos_simple(pred_tuple)
      targ_vos_tuple = pose_utils.calc_vos_simple(targ_tuple)

      s = pred_vos.size()
      vo_loss = \
        torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                              targ_vos.view(-1, *s[2:])[:, :3]) + \
        self.srx + \
        torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                              targ_vos.view(-1, *s[2:])[:, 3:]) + \
        self.srq

      vo_loss_tuple = \
        torch.exp(-self.srx) * self.t_loss_fn(pred_vos_tuple.view(-1, *s[2:])[:, :3],
                                              targ_vos_tuple.view(-1, *s[2:])[:, :3]) + \
        self.srx + \
        torch.exp(-self.srq) * self.q_loss_fn(pred_vos_tuple.view(-1, *s[2:])[:, 3:],
                                              targ_vos_tuple.view(-1, *s[2:])[:, 3:]) + \
        self.srq
      
      vo_loss = vo_loss + vo_loss_tuple
    # total pose loss

    pose_loss = abs_loss + vo_loss 

    # triplet loss + pose loss
    loss = torch.exp(-self.sc)*triplet_loss + self.sc + \
           torch.exp(-self.sp)*perceptual_loss + self.sp + \
           pose_loss

    loss_ = Loss(abs_loss=(t_loss,q_loss),vo_loss=vo_loss,triplet_loss=triplet_loss,perceptual_loss=perceptual_loss,final_loss=loss)
    return loss_