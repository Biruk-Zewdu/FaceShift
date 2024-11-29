#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.5, scale=64, **kwargs):
        """
        ArcFace Loss Function
        
        Args:
            nOut (int): Size of embedding features
            nClasses (int): Number of classes
            margin (float): Additive angular margin (default: 0.5)
            scale (float): Feature scale (default: 64)
        """
        super(LossFunction, self).__init__()
        
        self.test_normalize = True
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = True
        
        # Weight normalization
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)

        print('Initialised ArcFace Loss with margin %.3f and scale %.3f'%(margin, scale))

    def forward(self, x, label=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features
            label (torch.Tensor): Ground truth labels
        """
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine and sine of theta
        cos_theta = F.linear(x_norm, w_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        
        # Calculate cos(theta + margin)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.theta, cos_theta_m, cos_theta - self.sinmm)
        
        # Convert label to one-hot
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # Calculate output logits
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output = output * self.scale
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(output, label)
        
        return loss 