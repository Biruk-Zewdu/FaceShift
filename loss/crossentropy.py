#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, alpha=1.0, beta=1.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.alpha = alpha  # Weight for standard CE
        self.beta = beta    # Weight for reverse CE
        
        # Linear layer for projecting features to logits
        self.fc = nn.Linear(nOut, nClasses)
        
        print('Initialised Symmetric Cross Entropy Loss with alpha={:.2f}, beta={:.2f}'.format(alpha, beta))

    def forward(self, x, label=None):
        # Project features to logits
        x = self.fc(x)
        
        # Convert labels to one-hot encoding
        label_one_hot = F.one_hot(label, num_classes=x.size(-1)).float()
        
        # Compute probabilities
        pred = F.softmax(x, dim=1)
        
        # Standard cross entropy: -y * log(p)
        ce = -torch.sum(label_one_hot * F.log_softmax(x, dim=1), dim=1)
        
        # Reverse cross entropy: -p * log(y)
        # Add small epsilon to avoid log(0)
        rce = -torch.sum(pred * torch.log(label_one_hot + 1e-6), dim=1)
        
        # Combine both terms
        loss = self.alpha * ce + self.beta * rce
        
        return loss.mean() 