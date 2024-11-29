#! /usr/bin/python
# -*- encoding: utf-8 -*-

import timm

def MainModel(nOut=256, **kwargs):
    
    return timm.create_model('seresnet18', num_classes=nOut) 