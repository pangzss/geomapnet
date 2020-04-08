"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import sys
import os
import os.path as osp
import time
import configparser
import numpy as np
from visdom import Visdom
from shutil import copyfile
from common import Logger

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torch.cuda
from torch.autograd import Variable

from AdaIN import net
from AdaIN.function import adaptive_instance_normalization

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

def load_state_dict(model, state_dict):
  """
  Loads a state dict when the model has some prefix before the parameter names
  :param model: 
  :param state_dict: 
  :return: loaded model
  """
  model_names = [n for n,_ in model.named_parameters()]
  state_names = [n for n in list(state_dict.keys())]

  # find prefix for the model and state dicts from the first param name
  if model_names[0].find(state_names[0]) >= 0:
    model_prefix = model_names[0].replace(state_names[0], '')
    state_prefix = None
  elif state_names[0].find(model_names[0]) >= 0:
    state_prefix = state_names[0].replace(model_names[0], '')
    model_prefix = None
  else:
    print('Could not find the correct prefixes between {:s} and {:s}'.\
      format(model_names[0], state_names[0]))
    raise KeyError

  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in list(state_dict.items()):
    if state_prefix is None:
      k = model_prefix + k
    else:
      k = k.replace(state_prefix, '')
    new_state_dict[k] = v

  model.load_state_dict(new_state_dict)

def safe_collate(batch):
  """
  Collate function for DataLoader that filters out None's
  :param batch: minibatch
  :return: minibatch filtered for None's
  """
  batch = [x for x in batch if x is not None]
  return default_collate(batch)

class Trainer(object):
  def __init__(self, model, optimizer, train_criterion, config_file, experiment,
      train_dataset, val_dataset, device, seed=0, checkpoint_file=None, alpha=1,
      resume_optim=False, val_criterion=None,visdom_server='http://localhost', visdom_port=8097):
    """
    General purpose training script
    :param model: Network model
    :param optimizer: object of the Optimizer class, wrapping torch.optim
    and lr
    :param train_criterion: Training loss function
    :param config_file: configuration .ini file for training parameters
    :param experiment: name of the experiment, used to create logging dir
    :param train_dataset: PyTorch dataset
    :param val_dataset: PyTorch dataset
    :param device: IDs of the GPUs to use - value of $CUDA_VISIBLE_DEVICES
    :param checkpoint_file: Name of file with saved weights and optim params
    :param resume_optim: whether to resume optimization
    :param val_criterion: loss function to be used for validation
    """
    self.model = model
    self.train_criterion = train_criterion
    if val_criterion is None:
      self.val_criterion = self.train_criterion
    else:
      self.val_criterion = val_criterion
    self.alpha = alpha
    self.experiment = experiment
    self.optimizer = optimizer
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = device

    # read the config
    settings = configparser.ConfigParser()
    with open(config_file, 'r') as f:
      settings.read_file(f)
    self.config = {}

    section = settings['training']
    self.config['n_epochs'] = section.getint('n_epochs')
    self.config['batch_size'] = section.getint('batch_size')
    self.config['do_val'] = section.getboolean('do_val')
    self.config['shuffle'] = section.getboolean('shuffle')
    self.config['seed'] = section.getint('seed',seed)
    self.config['num_workers'] = section.getint('num_workers')
    self.config['snapshot'] = section.getint('snapshot')
    self.config['val_freq'] = section.getint('val_freq')
    self.config['cuda'] = torch.cuda.is_available()
    self.config['max_grad_norm'] = section.getfloat('max_grad_norm', 0)
    self.config['patience'] = section.getint('patience')

    section = settings['logging']
    self.config['log_visdom'] = section.getboolean('visdom')
    self.config['print_freq'] = section.getint('print_freq')

    self.logdir = osp.join(os.getcwd(), 'logs', self.experiment)
    i = 0
    while osp.isdir(self.logdir):
      i += 1
      temp = self.experiment + '_version_{}'.format(i)
      self.logdir = osp.join(os.getcwd(), 'logs', temp)
    self.experiment = self.experiment + '_version_{}'.format(i)
    os.makedirs(self.logdir)
    copyfile(config_file, osp.join(self.logdir, 'config.ini'))

    if self.config['log_visdom']:
      # start plots
      self.vis_env = experiment

      self.training_loss_win = 'training_loss_win'
      self.vis = Visdom(server=visdom_server, port=visdom_port)
      self.vis.line(X=np.zeros(1), Y=np.zeros(1), win=self.training_loss_win,
        opts={'legend': ['train_loss'], 'xlabel': 'epochs',
              'ylabel': 'loss'}, env=self.vis_env)
      
      self.val_loss_win = 'val_loss_win'
      self.vis.line(X=np.zeros(1), Y=np.zeros(1), win=self.val_loss_win,
        opts={'legend': ['val_loss'], 'xlabel': 'epochs',
              'ylabel': 'loss'}, env=self.vis_env)

      self.lr_win = 'lr_win'
      self.vis.line(X=np.zeros(1), Y=np.zeros(1), win=self.lr_win,
        opts={'legend': ['learning_rate'], 'xlabel': 'epochs',
              'ylabel': 'log(lr)'}, env=self.vis_env)
      criterion_params = {k: v.data.cpu().numpy()[0] for k, v in
                          self.train_criterion.named_parameters()}
      self.n_criterion_params = len(criterion_params)
      if self.n_criterion_params:
        self.criterion_param_win = 'cparam_win'
        self.vis.line(X=np.zeros((1, self.n_criterion_params)),
                      Y=np.asarray(list(criterion_params.values()))[np.newaxis, :],
                      win=self.criterion_param_win, env=self.vis_env,
                      opts={'legend': list(criterion_params.keys()),
                            'xlabel': 'epochs', 'ylabel': 'value'})

    logfile = osp.join(self.logdir, 'log_epoch{}.txt'.format(self.config['n_epochs']))
    stdout = Logger.Logger(logfile)
    print('Logging to {:s}'.format(logfile))
    sys.stdout = stdout

    # log all the command line options
    print('---------------------------------------')
    print('Experiment: {:s}'.format(self.experiment))
    for k, v in list(self.config.items()):
      print('{:s}: {:s}'.format(k, str(v)))
    print('Using GPU {:s} / {:d}'.format(device, torch.cuda.device_count()))
    print('seed: {}'.format(self.config['seed']))
    print('---------------------------------------')

    # set random seed
    torch.manual_seed(self.config['seed'])
    if self.config['cuda']:
      torch.cuda.manual_seed(self.config['seed'])

    self.start_epoch = int(0)
    if checkpoint_file:
      if osp.isfile(checkpoint_file):
        loc_func = None if self.config['cuda'] else lambda storage, loc: storage
        checkpoint = torch.load(checkpoint_file, map_location=loc_func)
        load_state_dict(self.model, checkpoint['model_state_dict'])
        if resume_optim:
          
          self.optimizer.learner.load_state_dict(checkpoint['optim_state_dict'])
          
          # the states are loaded into cpu. Convert them back to GPU to avoid errors
          for state in self.optimizer.learner.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

          self.start_epoch = checkpoint['epoch']
          if 'criterion_state_dict' in checkpoint:
            c_state = checkpoint['criterion_state_dict']
            append_dict = {k: torch.Tensor([0.0])
                           for k,_ in self.train_criterion.named_parameters()
                           if not k in c_state}
            c_state.update(append_dict)
            self.train_criterion.load_state_dict(c_state)
        print('Loaded checkpoint {:s} epoch {:d}'.format(checkpoint_file,
          checkpoint['epoch']))

    self.train_loader = torch.utils.data.DataLoader(train_dataset,
      batch_size=self.config['batch_size'], shuffle=self.config['shuffle'],
      num_workers=self.config['num_workers'], pin_memory=True,
      collate_fn=safe_collate)
    if self.config['do_val']:
      self.val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=self.config['batch_size'], shuffle=self.config['shuffle'],
        num_workers=self.config['num_workers'], pin_memory=True,
        collate_fn=safe_collate)
    else:
      self.val_loader = None

    # activate GPUs
    if self.config['cuda']:
      self.model.cuda()
      self.train_criterion.cuda()
      self.val_criterion.cuda()

  def save_checkpoint(self, epoch):
    filename = osp.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
    checkpoint_dict =\
      {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
       'optim_state_dict': self.optimizer.learner.state_dict(),
       'criterion_state_dict': self.train_criterion.state_dict()}
    torch.save(checkpoint_dict, filename)

  def train_val(self, lstm):
    """
    Function that does the training and validation
    :param lstm: whether the model is an LSTM
    :return: 
    """
    val_loss_list = []
    for epoch in range(self.start_epoch, self.config['n_epochs']):
      # VALIDATION

      if self.config['do_val'] and ((epoch % self.config['val_freq'] == 0) or
                                      (epoch == self.config['n_epochs']-1)) :
        val_batch_time = Logger.AverageMeter()
        val_loss = Logger.AverageMeter()
        self.model.eval()
        end = time.time()
        val_data_time = Logger.AverageMeter()
        for batch_idx, (data, target) in enumerate(self.val_loader):
          val_data_time.update(time.time() - end)

          kwargs = dict(target=target, criterion=self.val_criterion,
            optim=self.optimizer, train=False)
          if lstm:
            loss, _ = step_lstm(data, self.model, self.config['cuda'], **kwargs)
          else:
            loss, _ = step_feedfwd(data, self.model, self.config['cuda'],
              **kwargs)

          val_loss.update(loss)


          val_batch_time.update(time.time() - end)

          if batch_idx % self.config['print_freq'] == 0:
            print('Val {:s}: Epoch {:d}\t' \
                  'Batch {:d}/{:d}\t' \
                  'Data time {:.4f} ({:.4f})\t' \
                  'Batch time {:.4f} ({:.4f})\t' \
                  'Loss {:f}' \
              .format(self.experiment, epoch, batch_idx, len(self.val_loader)-1,
              val_data_time.val, val_data_time.avg, val_batch_time.val,
              val_batch_time.avg, loss))
            if self.config['log_visdom']:
              self.vis.save(envs=[self.vis_env])

          end = time.time()

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(self.experiment,
          epoch, val_loss.avg))
        val_loss_list.append(val_loss.avg)
        if self.config['log_visdom']:
          self.vis.line(X=np.asarray([epoch]),
            Y=np.asarray([val_loss.avg]), win=self.val_loss_win, name='val_loss',
            update='append', env=self.vis_env)
          self.vis.save(envs=[self.vis_env])

      # SAVE CHECKPOINT
      if epoch % self.config['snapshot'] == 0:
        if self.config['val_freq'] == self.config['snapshot']:
          if val_loss_list[-1] < min(val_loss_list[:-1]):
            self.save_checkpoint(epoch)
            print('Epoch {:d} checkpoint saved for {:s}'.\
               format(epoch, self.experiment))

          elif epoch == self.config['n_epochs'] - 1:
            self.save_checkpoint(epoch)
            print('Epoch {:d} checkpoint saved for {:s}'.\
              format(epoch, self.experiment))
        else:
          self.save_checkpoint(epoch)
          print('Epoch {:d} checkpoint saved for {:s}'.\
           format(epoch, self.experiment))

      # ADJUST LR
      lr = self.optimizer.adjust_lr(epoch)
      if self.config['log_visdom']:
        self.vis.line(X=np.asarray([epoch]), Y=np.asarray([np.log10(lr)]),
          win=self.lr_win, name='learning_rate', update='append', env=self.vis_env)

      # TRAIN
      self.model.train()
      train_data_time = Logger.AverageMeter()
      train_batch_time = Logger.AverageMeter()
      end = time.time()
      for batch_idx, (data, target) in enumerate(self.train_loader):
        train_data_time.update(time.time() - end)

        kwargs = dict(target=target, criterion=self.train_criterion,
          optim=self.optimizer, train=True,
          max_grad_norm=self.config['max_grad_norm'])
        if lstm:
          loss, _ = step_lstm(data, self.model, self.config['cuda'], **kwargs)
        else:
          loss, _ = step_feedfwd(data, self.model, self.config['cuda'],
            **kwargs)

        train_batch_time.update(time.time() - end)

        if batch_idx % self.config['print_freq'] == 0:
          n_iter = epoch*len(self.train_loader) + batch_idx
          epoch_count = float(n_iter)/len(self.train_loader)
          print('Train {:s}: Epoch {:d}\t' \
                'Batch {:d}/{:d}\t' \
                'Data Time {:.4f} ({:.4f})\t' \
                'Batch Time {:.4f} ({:.4f})\t' \
                'Loss {:f}\t' \
                'lr: {:f}'.\
            format(self.experiment, epoch, batch_idx, len(self.train_loader)-1,
            train_data_time.val, train_data_time.avg, train_batch_time.val,
            train_batch_time.avg, loss, lr))
          if self.config['log_visdom']:
            self.vis.line(X=np.asarray([epoch_count]),
              Y=np.asarray([loss]), win=self.training_loss_win, name='train_loss',
              update='append', env=self.vis_env)
            if self.n_criterion_params:
              for name, v in self.train_criterion.named_parameters():
                v = v.data.cpu().numpy()[0]
                self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([v]),
                                     win=self.criterion_param_win, name=name,
                                     update='append', env=self.vis_env)
            self.vis.save(envs=[self.vis_env])

        end = time.time()

    # Save final checkpoint
    epoch = self.config['n_epochs']
    self.save_checkpoint(epoch)
    print('Epoch {:d} checkpoint saved'.format(epoch))
    if self.config['log_visdom']:
      self.vis.save(envs=[self.vis_env])

  def style_train_val(self):
      """
      Function that does the training and validation
      :param lstm: whether the model is an LSTM
      :return: 
      """
      val_loss_list = []
      early_stop_counter = 0
      saved_list = []
      for epoch in range(self.start_epoch, self.config['n_epochs']):
        # VALIDATION
        if self.config['do_val'] and ((epoch % self.config['val_freq'] == 0) or
                                        (epoch == self.config['n_epochs']-1)) :
                      
          val_batch_time = Logger.AverageMeter()
          val_loss = Logger.AverageMeter()
          self.model.eval()
          end = time.time()
          val_data_time = Logger.AverageMeter()
          for batch_idx, (data, target) in enumerate(self.val_loader):

            imgs = data[0]
         
            #from common.vis_utils import show_batch, show_stereo_batch
            #from torchvision.utils import make_grid
            #show_batch(make_grid(imgs.reshape(-1,3,256,455), nrow=6, padding=5, normalize=False))
            #sys.exit(-1)

            val_data_time.update(time.time() - end)
            
            kwargs = dict(target=target, criterion=self.val_criterion,
              optim=self.optimizer, train=False)
      
            end = time.time()
            a_loss,r_loss, _ = step_feedfwd(imgs, self.model, self.config['cuda'],
              **kwargs)
            loss = a_loss+r_loss
            val_loss.update(loss)
            
            val_batch_time.update(time.time() - end)

            if batch_idx % self.config['print_freq'] == 0:
              print('Val {:s}: Epoch {:d}\t' \
                    'Batch {:d}/{:d}\t' \
                    'Data time {:.4f} ({:.4f})\t' \
                    'Step time {:.4f} ({:.4f})\t' \
                    'Loss {:f}' \
                .format(self.experiment, epoch, batch_idx, len(self.val_loader)-1,
                val_data_time.val, val_data_time.avg, val_batch_time.val,
                val_batch_time.avg, loss))
              if self.config['log_visdom']:
                self.vis.save(envs=[self.vis_env])

            end = time.time()

          print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(self.experiment,
            epoch, val_loss.avg))
          val_loss_list.append(val_loss.avg)

          if self.config['log_visdom']:
            self.vis.line(X=np.asarray([epoch]),
              Y=np.asarray([val_loss.avg]), win=self.val_loss_win, name='val_loss',
              update='append', env=self.vis_env)
            self.vis.save(envs=[self.vis_env])


         # SAVE CHECKPOINT
        if epoch % self.config['snapshot'] == 0 and len(val_loss_list)>=2:
          
          if self.config['val_freq'] == self.config['snapshot']:
            curr_val = val_loss_list[-1]
            past_best = min(val_loss_list[:-1])
            if curr_val < past_best:
              saved_list.append(epoch)
              if len(saved_list)>1 and epoch<300:
                os.remove(os.path.join('logs',self.experiment,'epoch_{:03d}.pth.tar'.format(saved_list[0])))
                print('epoch_{}.pth.tar deleted.'.format(saved_list[0]))
                del saved_list[0]
              #early stop
              if epoch > 300:
                early_stop_counter = 0
              print(f'Validation loss decreased ({past_best:.6f} --> {curr_val:.6f}). Zero counter and save model ...')
              self.save_checkpoint(epoch)
              print('Epoch {:d} checkpoint saved for {:s}'.\
                format(epoch, self.experiment))
            elif epoch == self.config['n_epochs'] - 1:
              self.save_checkpoint(epoch)
              print('Epoch {:d} checkpoint saved for {:s}'.\
                format(epoch, self.experiment))
            elif epoch > 300:
              early_stop_counter += self.config['val_freq']
              print('Early stop counter value: {}'.format(early_stop_counter))
              if early_stop_counter == self.config['patience']:
                print('Val error never decreases in {} epochs. Exit training.'.format(early_stop_counter))
                sys.exit(-1)
          elif epoch != 0 :
            self.save_checkpoint(epoch)
            print('Epoch {:d} checkpoint saved for {:s}'.\
            format(epoch, self.experiment))

        # ADJUST LR
  
        lr = self.optimizer.adjust_lr(epoch)
        if self.config['log_visdom']:
          self.vis.line(X=np.asarray([epoch]), Y=np.asarray([np.log10(lr)]),
            win=self.lr_win, name='learning_rate', update='append', env=self.vis_env)

        # TRAIN
        self.model.train()
        train_data_time = Logger.AverageMeter()
        train_batch_time = Logger.AverageMeter()
        end = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
          train_data_time.update(time.time() - end)
          # update batch
          data_shape = data[0].shape
          to_shape = (-1,data_shape[-3],data_shape[-2],data_shape[-1])
          real = data[0].reshape(to_shape)
          content_style = data[1]
          content = content_style[:,0].reshape(to_shape)
          style = content_style[:,1].reshape(to_shape)
          style_indc = data[2].view(-1)
          if sum(style_indc == 1) > 0:
              with torch.no_grad():
                  if self.alpha < 0:
                    self.alpha = np.random.rand(1).item()
                  content_f = vgg(content[style_indc == 1].cuda())
                  style_f = vgg(style[style_indc == 1].cuda())
                  feat = adaptive_instance_normalization(content_f, style_f)
                  feat = feat * self.alpha + content_f * (1 - self.alpha)
                  stylized = decoder(feat)
                  real[style_indc == 1] = stylized.cpu()
  
         # from common.vis_utils import show_batch, show_stereo_batch
         # from torchvision.utils import make_grid
         # show_batch(make_grid(real, nrow=6, padding=5, normalize=True))
         # sys.exit(-1)

          real = real.reshape(data_shape)
          kwargs = dict(target=target, criterion=self.train_criterion,
            optim=self.optimizer, train=True,
            max_grad_norm=self.config['max_grad_norm'])

      
          end = time.time()
          loss, _ = step_feedfwd(real, self.model, self.config['cuda'],
            **kwargs)
          
          train_batch_time.update(time.time() - end)

          if batch_idx % self.config['print_freq'] == 0:
            n_iter = epoch*len(self.train_loader) + batch_idx
            epoch_count = float(n_iter)/len(self.train_loader)
            print('Train {:s}: Epoch {:d}\t' \
                  'Batch {:d}/{:d}\t' \
                  'Data Time {:.4f} ({:.4f})\t' \
                  'Step Time {:.4f} ({:.4f})\t' \
                  'Loss {:f}\t' \
                  'lr: {:f}'.\
              format(self.experiment, epoch, batch_idx, len(self.train_loader)-1,
              train_data_time.val, train_data_time.avg, train_batch_time.val,
              train_batch_time.avg, loss, lr))
            if self.config['log_visdom']:
              self.vis.line(X=np.asarray([epoch_count]),
                Y=np.asarray([loss]), win=self.training_loss_win, name='train_loss',
                update='append', env=self.vis_env)
              if self.n_criterion_params:
                for name, v in self.train_criterion.named_parameters():
                  v = v.data.cpu().numpy()[0]
                  self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([v]),
                                      win=self.criterion_param_win, name=name,
                                      update='append', env=self.vis_env)
              self.vis.save(envs=[self.vis_env])

          end = time.time()
      # Save final checkpoint
      epoch = self.config['n_epochs']
      self.save_checkpoint(epoch)
      print('Epoch {:d} checkpoint saved'.format(epoch))
      if self.config['log_visdom']:
        self.vis.save(envs=[self.vis_env])

def step_feedfwd(data, model, cuda, target=None, criterion=None, optim=None,
    train=True, max_grad_norm=0.0):
      """
      training/validation step for a feedforward NN
      :param data: 
      :param target: 
      :param model: 
      :param criterion: 
      :param optim: 
      :param cuda: whether CUDA is to be used
      :param train: training / val stage
      :param max_grad_norm: if > 0, clips the gradient norm
      :return: 
      """
      if train:
        assert criterion is not None
      data_var = Variable(data, requires_grad=train)
      if cuda:
        data_var = data_var.cuda(non_blocking=True)
      with torch.set_grad_enabled(train):
        output = model(data_var)
      if criterion is not None:
        if cuda:
          target = target.cuda(non_blocking=True)
        target_var = Variable(target, requires_grad=False)
        with torch.set_grad_enabled(train):
          abs_loss,vo_loss = criterion(output, target_var)
          loss = abs_loss+vo_loss
        if train:
          # SGD step
          optim.learner.zero_grad()
          loss.backward()
          if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
          optim.learner.step()
          return loss.item(), output
        else:
          return abs_loss.item(),vo_loss.item(),output
      else:
        return 0, output


def step_lstm(data, model, cuda, target=None, criterion=None, optim=None,
    train=True):
  """
  NEVER USED
  training/validation step for a feedforward NN
  :param data: N x T x C x H x w
  :param target: N x T x 7
  :param model: 
  :param criterion: 
  :param optim: instance of Optimizer
  :param cuda: whether CUDA is to be used
  :param train: training / val stage
  :return: 
  """
  # favor BPTT over batch size
  M = 64  # no. of images that can fit on the GPU
  if not train:
    M *= 2
  N, T = data.size(0), data.size(1)
  G = min(T, M)  # no. of time slices that can fit on the GPU
  B = min(N, M/G)  # batch size that can fit on the GPU

  if train:
    assert criterion is not None

  data_var = Variable(data, volatile=(not train), requires_grad=train)

  loss_accum = 0
  b_start = np.random.randint(N%B + 1)
  for b in range(N/B):
    b_idx = b_start + torch.LongTensor(range(b*B, (b+1)*B))
    xb = torch.index_select(data_var, dim=0, index=Variable(b_idx))
    if target is not None:
      tb = torch.index_select(target, dim=0, index=Variable(b_idx).cuda())
    model.reset_hidden_states(B)
    g_start = np.random.randint(T%G + 1)
    for g in range(T/G):
      g_idx = g_start + torch.LongTensor(range(g*G, (g+1)*G))
      xg = torch.index_select(xb, dim=1, index=Variable(g_idx))
      if target is not None:
        tg = torch.index_select(tb, dim=1, index=Variable(g_idx).cuda())
      model.detach_hidden_states()
      output = model(xg, cuda=cuda, non_blocking=True)

      if criterion is not None:
        if cuda:
          tg = tg.cuda(non_blocking=True)
        tg_var = Variable(tg, volatile=(not train), requires_grad=False)
        loss = criterion(output, tg_var)
        loss_accum += loss.data[0]

        if train:
          # SGD step
          optim.learner.zero_grad()
          loss.backward()
          optim.learner.step()

  return loss_accum, output
