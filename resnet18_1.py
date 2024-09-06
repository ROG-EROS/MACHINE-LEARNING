# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:10:27 2023

@author: nagal
"""

import torchvision.models as models
resnet18=models.googlenet(pretrained="True")
print(resnet18.eval())