#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:59:49 2019

@author: best
"""

import torch
import sys
import math
import numpy as np
from utils import configs
from utils.ops import load_image
from PIL import Image
import os
import scipy
from scipy import io

import matplotlib.pyplot as plt
import time
import copy
import torchvision
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parameter import Parameter
sys.dont_write_bytecode = True

'''
pos_x, pos_y = np.meshgrid(
                np.linspace(1, 109, 109),
                np.linspace(1, 109, 109)
                )

x = pos_x.reshape(109*109)
y = pos_y.reshape(109*109)
'''

test_img = np.array(Image.open('/media/best/A_Coding_Disk/sae_ws/test_image/00072.jpg'))
pixel_x1 = int(0.6944444 * 300)    
pixel_y1 = int(0.8333333 * 300)
pixel_x2 = int(0.88004136 * 300)
pixel_y2 = int(0.7075707 * 300)
     
test_img[pixel_x1][pixel_y1] = [0, 255, 0]
test_img[pixel_x1][pixel_y1 + 1] = [0, 255, 0]
test_img[pixel_x1][pixel_y1 - 1] = [0, 255, 0]
test_img[pixel_x1 - 1][pixel_y1] = [0, 255, 0]
test_img[pixel_x1 - 1][pixel_y1 + 1] = [0, 255, 0]
test_img[pixel_x1 - 1][pixel_y1 - 1] = [0, 255, 0]
test_img[pixel_x1 + 1][pixel_y1] = [0, 255, 0]
test_img[pixel_x1 + 1][pixel_y1 + 1] = [0, 255, 0]
test_img[pixel_x1 + 1][pixel_y1 - 1] = [0, 255, 0]

test_img[pixel_x2][pixel_y2] = [255, 255, 0]
test_img[pixel_x2][pixel_y2 + 1] = [255, 255, 0]
test_img[pixel_x2][pixel_y2 - 1] = [255, 255, 0]
test_img[pixel_x2 + 1][pixel_y2] = [255, 255, 0]
test_img[pixel_x2 + 1][pixel_y2 + 1] = [255, 255, 0]
test_img[pixel_x2 + 1][pixel_y2 - 1] = [255, 255, 0]
test_img[pixel_x2 - 1][pixel_y2] = [255, 255, 0]
test_img[pixel_x2 - 1][pixel_y2 + 1] = [255, 255, 0]
test_img[pixel_x2 - 1][pixel_y2 - 1] = [255, 255, 0]

res = Image.fromarray(test_img)
res.save('/media/best/A_Coding_Disk/sae_ws/test_res/72.jpg')


