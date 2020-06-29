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

##############################################
###   Loading First GoogLeNet Conv Layer   ###
##############################################

# googlenet = models.inception(pretrained=True)

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

###########################
###   Tensor To Image   ###
###########################
def tensor_to_np(tensor):
    img = tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor.view([60, 60])
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

###############################
###   Spatial AutoEncoder   ###
###############################
    

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
        # self.register_buffer('pos_x', self.pos_x)
        # self.register_buffer('pos_y', self.pos_y)

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
        # print(encoded)
        decoded = self.decoder(encoded)
        decoded = decoded.view(50, 60, 60)
        
        encoded_last = self.encoder(x_last)
        encoded_next = self.encoder(x_next)
        
        #########################
        ###   Loss Function   ###
        #########################
        loss_func = torch.nn.MSELoss(reduce=False, size_average=False)
        gslow = loss_func((encoded - encoded_next), (encoded - encoded_last))
        loss = torch.sum(loss_func(decoded, torch.tensor(x_downsample))) + torch.sum(gslow)
        return loss, decoded[0], encoded
    
    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()
        
net = SAE()


##########################
###   Loading Images   ###
##########################
transform = transforms.Compose([
    transforms.ToTensor()
])

class ImportData(data.Dataset):
    def __init__(self,root):
        imgs = os.listdir(root)
        imgs = sorted(imgs)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms = transform

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
        return img

    def __len__(self):
        return len(self.imgs)

class ImportFloat(data.Dataset):
    def __init__(self,root):
        imgs = os.listdir(root)
        imgs = sorted(imgs)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms = transform

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
        return np.float32(img / 255)

    def __len__(self):
        return len(self.imgs)

block_dataset = ImportData('/media/best/A_Coding_Disk/sae_ws/jpeg_file2')
block_float = ImportFloat('/media/best/A_Coding_Disk/sae_ws/jpeg_file2')
############################
###   Choosing Batches   ###
############################

def new_trail() :
    # Creating a list of 50 list
    num1 = [] # store 0 to 99 in num1 list to represent images from 1 to 100
    k = 0
    for i in range(100):
        num1.append(k)
        k += 1
    return num1        

def init_num0() :
    num1 = new_trail()
    # store 50 list of num1 in num0
    num0 = [num1] * 50
    for i in range(50) :
        num0[i] = new_trail()
    return num0

def get_batch_data(num0) :
    batch_xs = []
    batch_xs_last = []
    batch_xs_next = []
    batch_xs_img = []
    for i in range(len(num0)):  # form 1st to 50th list(image trail) in num0
        temp = np.random.randint(0, len(num0[i]))  # get a random image from 100 images
        batch_xs.append(block_float[ i * len(num0) + num0[i][temp]])  # append 1 random image from each trail

        batch_xs_img.append(
                   np.float32(np.array(
                   Image.fromarray(
                   block_dataset[ i * len(num0) + num0[i][temp]].transpose(1, 2, 0)
                   ).resize((60, 60), Image.ANTIALIAS)
                   .convert('L') ) / 255))
       
        if temp == 0:  # if there's no last, compute 2*next
            batch_xs_next.append(block_float[ i * len(num0) + num0[i][temp] + 1])
            batch_xs_last.append(block_float[i * len(num0) + num0[i][temp] + 1])
        elif temp == len(num0[i]) - 1:  # if there's no next, compute 2*last 
            batch_xs_last.append(block_float[ i * len(num0) + num0[i][temp] - 1])
            batch_xs_next.append(block_float[i * len(num0) + num0[i][temp] - 1])
        else :
            batch_xs_last.append(block_float[i * len(num0) + num0[i][temp] - 1])
            batch_xs_next.append(block_float[i * len(num0) + num0[i][temp] + 1])
        num0[i].pop(temp)  # delete this image index from the trail
    return batch_xs, batch_xs_last, batch_xs_next, batch_xs_img

##########################
### Train Parameters   ###
##########################
epochs = 20
batch_size = 50
num_batch = int(5000 / 50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(net.parameters(), lr = 0.0005)
num0 = init_num0()

# batch_xs, batch_xs_last, batch_xs_next, batch_xs_img = get_batch_data(num0)

#######################
###   Main Function ###
#######################
net.to(device)

for i in range(epochs):
    num0 = init_num0()
    for j in range(num_batch):
        batch_xs, batch_xs_last, batch_xs_next, batch_xs_img= get_batch_data(num0)
        batch_xs = torch.tensor(batch_xs)
        batch_xs_last = torch.tensor(batch_xs_last).detach()
        batch_xs_next = torch.as_tensor(batch_xs_next).detach()
        loss, decoded, feature = net.train(batch_xs, batch_xs_last, batch_xs_next, batch_xs_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (j + 1) % 10 == 0:
            print('Epoch: {}, Batch_num: {}, Loss: {:.4f}'.format(i, j, loss))
            # points = feature[0].detach()
            # print(points)
            image_decoded = batch_xs_img[0]
            image_gray = decoded.detach()
            plt.subplot(121); plt.imshow(image_gray, cmap = 'gray')
            plt.subplot(122); plt.imshow(image_decoded, cmap = 'gray')
            filename = '/media/best/A_Coding_Disk/sae_ws/pytorch_output/' + str('%03d' % i) + str('%03d' % j) + '.jpg'
            plt.savefig(filename)
        if((j + 1) % num_batch == 0) or j == 0:
            torch.save(net.state_dict(), '/media/best/A_Coding_Disk/sae_ws/pytorch_model/sae.pkl')
            
