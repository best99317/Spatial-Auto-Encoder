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


file_name = '/media/best/Coding Disk/sae_ws/inception_v1.ckpt' #.ckpt的路径
name_variable_to_restore = 'InceptionV1/Conv2d_1a_7x7/weights' #要读取权重的变量名
name_variable2_to_restore = 'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta'
reader = pywrap_tensorflow.NewCheckpointReader(file_name)
var_to_shape_map = reader.get_variable_to_shape_map()
'''
for key in var_to_shape_map:
    print("tensor_name: ", key)
'''
conv1 = tf.get_variable("Conv2d_1a_7x7", var_to_shape_map[name_variable_to_restore], trainable=False) # 定义接收权重的变量名
bias1 = tf.get_variable("Conv2d_1a_7x7/BatchNorm", var_to_shape_map[name_variable2_to_restore], trainable=False)

print(conv1)
print(bias1)

restorer_fc = tf.train.Saver({name_variable_to_restore: conv1 }) #定义恢复变量的对象
restorer_fc2 = tf.train.Saver({name_variable2_to_restore: bias1 })
sess = tf.Session()
sess.run(tf.variables_initializer([conv1], name='init')) #必须初始化
sess.run(tf.variables_initializer([bias1], name='init'))
restorer_fc.restore(sess, file_name) #恢复变量
restorer_fc2.restore(sess, file_name)
print(sess.run(conv1)) 
print(sess.run(bias1))