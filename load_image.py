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

image_raw = tf.gfile.FastGFile('/media/best/A_Coding_Disk/sae_ws/jpeg_file/Sequence 040000.jpg','rb').read()  #bytes
img = tf.image.decode_jpeg(image_raw) #Tensor
#img2 = tf.image.convert_image_dtype(img, dtype = tf.uint8)


img = Image.open('/media/best/A_Coding_Disk/sae_ws/jpeg_file/Sequence 040000.jpg')
img = np.array(img)
if img.ndim == 3:
    img = img[:,:,0]
plt.subplot(221); plt.imshow(img)
plt.subplot(222); plt.imshow(img, cmap ='gray')
plt.subplot(223); plt.imshow(img, cmap = plt.cm.gray)
plt.subplot(224); plt.imshow(img, cmap = plt.cm.gray_r)
plt.show()


with tf.Session() as sess:
  print(type(image_raw)) # bytes
  print(type(img)) # Tensor
  #print(type(img2))
 
  print(type(img.eval())) # ndarray !!!
  print(img.eval().shape)
  print(img.eval().dtype)
 
#  print(type(img2.eval()))
#  print(img2.eval().shape)
#  print(img2.eval().dtype)
  plt.figure(1)
  plt.imshow(img.eval())
  plt.show()