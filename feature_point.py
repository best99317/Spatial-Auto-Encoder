# -*- coding: utf-8 -*-
#
# The XJTU License (Xi'an Jiaotong University)
# Copyright (c) 2019 Site Bai.
# 
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


googlenet = io.loadmat('/media/best/A_Coding_Disk/sae_ws/imagenet-googlenet-dag.mat')
conv1_ = googlenet['params'][0][0][1] # 7 * 7 * 3 * 64


bias1_ = googlenet['params'][0][1][1]
bias1_pre = []
for i in range(len(bias1_)):
    bias1_pre.append(bias1_[i][0])



conv1_pre = np.zeros(64*3*7*7, dtype = np.float32)
for i1 in range(64):
    for i2 in range(3):
        for i3 in range(7):
            for i4 in range(7):
                conv1_pre[i1 *64 + i2 * 3 + i3 * 7 + i4] = conv1_[i3][i4][i2][i1]

conv1_pre = conv1_pre.reshape(64, 3, 7, 7)


pos_x, pos_y = np.meshgrid(
        np.linspace(0, 1, 109),
        np.linspace(0, 1, 109)
        )

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.


        self.pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        self.pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        # self.register_buffer('pos_x', pos_x)
        # self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2) # 3 * 64 * 7 * 7 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.spatial_softmax = SpatialSoftmax(109, 109, 16)
        self.fc = nn.Linear(32, 60 * 60)
        
        self.conv1.weight.data.copy_(torch.tensor(conv1_pre, dtype = torch.float))
        self.conv1.bias.data.copy_(torch.tensor(bias1_pre, dtype = torch.float))       
        
        
      
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.spatial_softmax.forward(x)
    
    def decoder(self, x):
        return self.fc(x)

    def train(self, x, x_last, x_next, x_downsample):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(1, 60, 60)

        encoded_last = self.encoder(x_last)
        encoded_next = self.encoder(x_next)
        
        #########################
        ###   Loss Function   ###
        #########################
        loss_func = torch.nn.MSELoss(reduce=False, size_average=False)
        gslow = loss_func((encoded - encoded_next), (encoded - encoded_last))
        loss = torch.sum(loss_func(decoded, torch.tensor(x_downsample))) + torch.sum(gslow)
        return loss, decoded[0]
    
    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()



pre_trained_net = SAE()
pre_trained_net.load_state_dict(torch.load('/media/best/A_Coding_Disk/sae_ws/pytorch_model/sae.pkl'))

class ImportData(data.Dataset):
    def __init__(self,root):
        imgs = os.listdir(root)
        imgs = sorted(imgs)
        self.imgs=[os.path.join(root,k) for k in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        '''
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        
        img = np.zeros(3 * 240 * 240, dtype = np.float32)
        '''
        pil_img = pil_img.resize((240, 240), Image.ANTIALIAS)
        pil_img = np.array(pil_img) 
        img = pil_img.transpose(2, 0, 1)
        return np.float32(img / 255 * 255)

    def __len__(self):
        return len(self.imgs)


block_dataset = ImportData('/media/best/A_Coding_Disk/sae_ws/test_image')
# test_img = [block_dataset[37]]
img = [block_dataset[25]]
img_last = [block_dataset[24]]
img_next = [block_dataset[26]]

# test_data = torch.tensor(block_dataset)
# feature_tensor = pre_trained_net.encoder(torch.tensor(test_img))
# feature_pos = feature_tensor.detach().numpy()

img_downsample = img[0]
img_downsample = img_downsample.transpose(1, 2, 0)
img_downsample = Image.fromarray(np.uint8(img_downsample))
img_downsample = img_downsample.resize((60, 60), Image.ANTIALIAS).convert('L')
img_downsample = [np.float32(np.array(img_downsample) / 255)]


loss, dec_img = pre_trained_net.train(torch.tensor(img), torch.tensor(img_last), torch.tensor(img_next), torch.tensor(img_downsample))
res = dec_img.detach()

plt.plot() 
plt.imshow(res, cmap = 'gray')



