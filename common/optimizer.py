"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np 
import torch.optim as optim
from common.scheduler import LearningRateScheduler

class Optimizer:
  """
  Wrapper around torch.optim + learning rate
  """
  def __init__(self, params, method, base_lr, weight_decay, **kwargs):
    self.method = method
    self.base_lr = base_lr
    self.decay = False
    self.schedule = False
    if 'lr_decay' in kwargs:
        self.decay = True
        self.lr_decay = kwargs.pop('lr_decay')
        self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
    elif 'end_lr' in kwargs:
        self.epochs = kwargs.pop('epochs')
        self.schedule = True
        self.scheduler = LearningRateScheduler(self.epochs, np.log10(base_lr), np.log10(kwargs.pop('end_lr')))
    assert(not (self.decay and self.schedule), 'Cannot have decay and schedule')

    if self.method == 'sgd':
      self.lr_decay = kwargs.pop('lr_decay')
      self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
      self.learner = optim.SGD(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'adam':
      self.learner = optim.Adam(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'rmsprop':
      self.learner = optim.RMSprop(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)

  def adjust_lr(self, epoch):
    if not (self.decay or self.schedule): #self.method != 'sgd':
      return self.base_lr

    if self.decay:
      decay_factor = 1
      for s in self.lr_stepvalues:
          if epoch < s:
              break
          decay_factor *= self.lr_decay

      lr = self.base_lr * decay_factor
    elif self.schedule:

      lr = self.scheduler.get_lr(epoch if epoch < self.epochs else self.epochs-1)



    for param_group in self.learner.param_groups:
        param_group['lr'] = lr

    return lr
  def mult_lr(self, f):
    for param_group in self.learner.param_groups:
      param_group['lr'] *= f
