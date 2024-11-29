#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
from torchvision.models.densenet import DenseNet

def MainModel(nOut=256, **kwargs):
    # Using growth_rate=28 instead of default 32
    # This will reduce parameters while maintaining the architecture pattern
    model = DenseNet(
        growth_rate=28,
        block_config=(6, 12, 32, 32),  # DenseNet-169 configuration
        num_init_features=64,
        num_classes=nOut
    )
    
    return model