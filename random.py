# -*- coding: utf-8 -*-
#
# The XJTU License (Xi'an Jiaotong University)
# Copyright (c) 2019 Site Bai.
# 


import sys
import math
import numpy as np
import tensorflow as tf
from utils import configs
from utils.ops import load_image
from PIL import Image
import os
import scipy
from scipy import io
from tensorflow.python import pywrap_tensorflow
import matplotlib.pyplot as plt

num1 = []

for i in range(50):
    num1.append(i)

num0 = []

for i in range(50):
    num0.append(num1)
    
temp = np.random.randint(0,49)