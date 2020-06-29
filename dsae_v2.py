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
import datetime

sys.dont_write_bytecode = True
#from tf_util import Convolution2D
#from tf_util import FullyConnected


##############################################
###   Loading First GoogLeNet Conv Layer   ###
##############################################

file_name = '/media/best/A_Coding_Disk/sae_ws/inception_v1.ckpt' 
name_conv_to_restore = 'InceptionV1/Conv2d_1a_7x7/weights' 
name_bias_to_restore = 'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta'
reader = pywrap_tensorflow.NewCheckpointReader(file_name)
var_to_shape_map = reader.get_variable_to_shape_map()

print("Start Loading GoogLeNet!")

conv1 = tf.get_variable("Conv2d_1a_7x7", var_to_shape_map[name_conv_to_restore], trainable=False)
bias1 = tf.get_variable("Conv2d_1a_7x7/BatchNorm", var_to_shape_map[name_bias_to_restore], trainable=False)
print(conv1.dtype)

'''
restorer_fc = tf.train.Saver({name_variable_to_restore: conv1}) 
sess = tf.Session()
sess.run(tf.variables_initializer([conv1], name='init')) 
restorer_fc.restore(sess, file_name) 
print(sess.run(conv1)) 
'''

########################################
###   Parts of Spatial AutoEncoder   ###
########################################

class Convolution_Pre :
    def __init__(self,in_ch,out_ch,ksize,stride) :
        print("!!!Network Initialized!!!")
        self.stride = stride
        self.w = conv1
        self.b = bias1
    
    def linear(self,x) :
        return tf.nn.conv2d( x, self.w, strides=[1, self.stride, self.stride, 1], padding="SAME" ) + self.b

    def relu(self,x) :
        return tf.nn.relu( self.linear(x) )



class Convolution2D :
    def __init__(self,in_ch,out_ch,ksize,stride) :
        print("!!!Network Initialized!!!")
        self.stride = stride
        self.w = tf.Variable( tf.truncated_normal([ksize,ksize,in_ch,out_ch]) )
        self.b = tf.Variable( tf.truncated_normal([out_ch]) )
        print(self.w.dtype)
    
    def linear(self,x) :
        return tf.nn.conv2d( x, self.w, strides=[1, self.stride, self.stride, 1], padding="SAME" ) + self.b

    def relu(self,x) :
        return tf.nn.relu( self.linear(x) )
    
class FullyConnected :
    def __init__(self,n_in,n_out) :
        print("!!!Network Initialized!!!")
        self.w = tf.Variable( tf.truncated_normal([n_in,n_out],stddev=0.0001) )
        self.b = tf.Variable( tf.truncated_normal([n_out],stddev=0.0001) )
        print(self.w.dtype)
        
    def linear(self,x) :
        return tf.matmul(x,self.w) + self.b 
    
    def relu(self,x) :
        return tf.nn.relu( self.linear(x) )

'''
class SpatialSoftmax :
    def __init__( self, shape = [-1,80,80,16] ) :
        self.shape = shape
        np_fe_x = np.zeros( (shape[1]*shape[2]*shape[3],shape[3]*2) ,dtype=np.float32 )
        np_fe_y = np.zeros( (shape[1]*shape[2]*shape[3],shape[3]*2) ,dtype=np.float32 )
            
        for y in range(shape[1]):
            for x in range(shape[2]):    
                for t in range(shape[3]):   
                    np_fe_x[ y*(shape[2]*shape[3]) + x*shape[3] + t ][t*2+1] = x +1 
                    np_fe_y[ y*(shape[2]*shape[3]) + x*shape[3] + t ][t*2  ] = y +1
        
        self.fe_x = tf.constant(np_fe_x)
        self.fe_y = tf.constant(np_fe_y)

    def act(self, x) :
        # convert to [-1, ch, height, width]
        trans = tf.transpose(x,perm=[0, 3, 1, 2])  
        
        # tf.nn.softmax
        dist_to_batch = tf.reshape(  trans, [-1, self.shape[1]*self.shape[2]] )
        print(dist_to_batch.eval().shape)
        spatial_softmax = tf.nn.softmax( dist_to_batch )
        print(spatial_softmax.eval().shape)
        batch_to_dist = tf.reshape( spatial_softmax, [-1, self.shape[3], self.shape[1], self.shape[2]] )
        print(batch_to_dist.eval().shape)
        # convert to [-1, height, width , ch ]
        distributed = tf.transpose( batch_to_dist, perm=[0,2,3,1] )
        softmax_out = tf.reshape( distributed , [-1, self.shape[1]*self.shape[2]*self.shape[3]]  )
        print(softmax_out.eval().shape)
        dy = float(self.shape[1])
        dx = float(self.shape[2])
        print(dy)
        # feature_points
        # TODO
        return (tf.matmul( softmax_out, self.fe_y ) ) + (tf.matmul( softmax_out, self.fe_x ) )
'''

class SpatialSoftmax:
    def __init__(self, features) :
        self.features = features
    
    def act(self, x) :
        return tf.contrib.layers.spatial_softmax(x)

###############################
###   Spatial AutoEncoder   ###
###############################

class DeepSpatialAutoEncoder:
    def __init__(
            self,
            input_shape = [-1,240,240,3],
            reconstruction_shape = [-1,60,60,1],
            filter_chs = [64,32,16],
            filter_sizes = [7,5,5],
            filter_strides = [2,1,1]
            ):
        self.filter_num = len(filter_chs) 
        if any(len(lst) != self.filter_num for lst in [filter_chs,filter_sizes,filter_strides]):
            raise NameError("size error.")
        
        
        self.input_shape = input_shape
        self.reconstruction_shape = reconstruction_shape

        self.filter_chs = filter_chs
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides

        self.convs = []
        self.shapes = [] # shape[0] -> conv1 -> shape[1] -> conv2 -> shape[2] -> ...
        self.shapes.append(input_shape)
        
        for i,(fch,fsize,fstride) in enumerate(zip(filter_chs,filter_sizes,filter_strides)) :
            in_shape = self.shapes[-1] #the last element in shapes
            if i == 0 :
                conv = Convolution_Pre( in_ch=in_shape[3],out_ch=fch,ksize=fsize,stride=fstride)
            else :
                conv = Convolution2D( in_ch=in_shape[3],out_ch=fch,ksize=fsize,stride=fstride)
            self.convs.append(conv)
            out_height = int( math.ceil(float(self.shapes[-1][1]) / float(fstride)) )
            out_width = int( math.ceil(float(self.shapes[-1][2]) / float(fstride)) )
            '''
            out_height = int( int(self.shapes[-1][1] - int(math.floor(float(fsize) / 2)) * 2) / fstride) 
            out_width = int( int(self.shapes[-1][2] - int(math.floor(float(fsize) / 2)) * 2) / fstride)
            '''
            self.shapes.append([-1,out_height,out_width,fch])
        # endof for

       
        self.spatial_softmax = SpatialSoftmax( self.shapes[-1] )
        self.fully_connected = FullyConnected( self.filter_chs[-1]*2, reconstruction_shape[1]*reconstruction_shape[2] )

    def add_encode(self, x) :
        i_input = x
        for i in range(len(self.convs)) :
            h = self.convs[i].relu( x=i_input )
            i_input = h
        # endof for
        h = self.spatial_softmax.act( i_input )
        return h

    def add_decode(self, x) :
        return self.fully_connected.linear(x)
    '''
    def add_train_without_gslow(self, x) :
        encoded = self.add_encode(x)
        decoded = self.add_decode(encoded)
        decoded_img = tf.reshape(decoded, [-1, self.reconstruction_shape[1],self.reconstruction_shape[2],1]) 

        sv = tf.image.rgb_to_grayscale( x )
        sv = tf.image.resize_images( sv, [self.reconstruction_shape[1],self.reconstruction_shape[2]],tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        sv_flatten = tf.reshape(sv, [-1, self.reconstruction_shape[1]*self.reconstruction_shape[2]])

        loss = tf.reduce_sum( tf.square( sv_flatten - decoded ) )
        
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(0.1, step, 1000, 0.96)
        optimizer = tf.train.AdamOptimizer(rate)
        targs = []
        for i in range(len(self.convs)) :
            targs.append(self.convs[i].w)
            targs.append(self.convs[i].b)
        targs.append(self.fully_connected.w)
        targs.append(self.fully_connected.b)

        # grads, variables = zip(*optimizer.compute_gradients( loss, targs ))
        # grads, _ = tf.clip_by_global_norm(grads, 1.25)
        # train_step = optimizer.apply_gradients(zip(grads, variables))
        grads = optimizer.compute_gradients(loss, targs)
        train_step = optimizer.apply_gradients(grads)
        return train_step,loss,decoded_img,sv
    '''
    
    def add_train(self, x, x_last, x_next) :
        encoded = self.add_encode(x)
        decoded = self.add_decode(encoded)
        decoded_img = tf.reshape(decoded, [-1, self.reconstruction_shape[1],self.reconstruction_shape[2],1]) 

        encoded_next = self.add_encode(x_next)
        encoded_next = tf.stop_gradient(encoded_next)
        encoded_last = self.add_encode(x_last)
        encoded_last = tf.stop_gradient(encoded_last)
        
        i_downsamp = tf.image.rgb_to_grayscale( x )
        i_downsamp = tf.image.resize_images( i_downsamp, [self.reconstruction_shape[1],self.reconstruction_shape[2]],tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        i_downsamp_flatten = tf.reshape(i_downsamp, [-1, self.reconstruction_shape[1]*self.reconstruction_shape[2]])


        gslow = tf.square( (encoded_next-encoded) - (encoded - encoded_last)) 
        loss = tf.reduce_sum( tf.square( i_downsamp_flatten - decoded ) ) + tf.reduce_sum( gslow )
        
        
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(0.0005, step, 1, 0.9999)
        optimizer = tf.train.AdamOptimizer(rate)
        targs = []
        for i in range(len(self.convs)) :
            targs.append(self.convs[i].w)
            targs.append(self.convs[i].b)
        targs.append(self.fully_connected.w)
        targs.append(self.fully_connected.b)

        # grads, variables = zip(*optimizer.compute_gradients( loss, targs ))
        # grads, _ = tf.clip_by_global_norm(grads, 1.25)
        # train_step = optimizer.apply_gradients(zip(grads, variables))
        grads = optimizer.compute_gradients(loss, targs)
        train_step = optimizer.apply_gradients(grads)
        return train_step,loss,decoded_img,i_downsamp,targs

##############################
###   Define Placeholder   ###
##############################
x = tf.placeholder(tf.float32, [None, 240, 240, 3]) 
x_last = tf.placeholder(tf.float32, [None, 240, 240, 3])       
x_next = tf.placeholder(tf.float32, [None, 240, 240, 3])  
    

###############################
###   Training Parameters   ###
###############################
training_epochs = 20
# batch_size = 1
batch_size = 50
display_step = 5
input_shape = [-1,240,240,3]
reconstruction_shape = [-1,60,60,1]
filter_chs = [64,32,16]
filter_sizes = [7,5,5]
filter_strides = [2,1,1]





##########################
###   Importing Data   ###
##########################

# set the directory of the image
image_path = '/media/best/A_Coding_Disk/sae_ws/jpeg_file1'

'''
def load__image(filename):
  with open( filename ) as f:
    return f.read()
'''

def preprocess_image(image):
    image_raw = tf.gfile.GFile(image,'rb').read()
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image, [240, 240])
    image /= 255.0
    return image

'''
def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

'''

# load data into "images"
print("Start Loading Path!")
image_names = [os.path.join(os.getcwd(), image_path, x)
                     for x in os.listdir(image_path) if x.endswith('.jpg')]

image_names = sorted(image_names)
# default_image = preprocess_image('sae_ws/jpeg_file/Sequence 040000.jpg')

# images = [default_image] * 5000

# for image_name in range(len(image_names)):
#    images[image_name] = preprocess_image(image_names[image_name])

'''
def get_batch_data(i, images):
    batch_xs = [default_image] * 50
    for j in range(50) :
        batch_xs[j] = preprocess_image(image_names[i * 50 + j])
    return batch_xs
'''


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
    for i in range(len(num0)):  # form 1st to 50th list(image trail) in num0
        temp = np.random.randint(0, len(num0[i]))  # get a random image from 100 images
        batch_xs.append(preprocess_image(image_names[ i * len(num0) + num0[i][temp]]))  # append 1 random image from each trail
        if temp == 0:  # if there's no last, compute 2*next
            batch_xs_next.append(preprocess_image(image_names[ i * len(num0) + num0[i][temp] + 1]))
            batch_xs_last.append(preprocess_image(image_names[i * len(num0) + num0[i][temp] + 1]))
        elif temp == len(num0[i]) - 1:  # if there's no next, compute 2*last 
            batch_xs_last.append(preprocess_image(image_names[ i * len(num0) + num0[i][temp] - 1]))
            batch_xs_next.append(preprocess_image(image_names[i * len(num0) + num0[i][temp] - 1]))
        else :
            batch_xs_last.append(preprocess_image(image_names[i * len(num0) + num0[i][temp] - 1]))
            batch_xs_next.append(preprocess_image(image_names[i * len(num0) + num0[i][temp] + 1]))
        num0[i].pop(temp)  # delete this image index from the trail
    return batch_xs, batch_xs_last, batch_xs_next
  
# print(images[0])
#######################
###   TF Sessions   ###
#######################

# Define a SAE
# Need to Pre-Define the Corresponding-Size Filters !!!! 

print("Start Initializing SAE")

SAE = DeepSpatialAutoEncoder(input_shape, reconstruction_shape, 
                             filter_chs, filter_sizes, filter_strides)

print("Start TF Session")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Graph().device('/gpu:0'), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(len(image_names) / batch_size)
    
    for epoch in range(training_epochs):
    # for epoch in range(1):
        num0 = init_num0()
        for i in range(total_batch):
        # for i in range(1):
            start1 = datetime.datetime.now()
            ######################   REAL DATA   ########################
            batch_xs, batch_xs_last, batch_xs_next = get_batch_data(num0)
            print(batch_xs[0].shape)
            print(batch_xs[0].dtype)
            ######################   REAL DATA   ########################
            end1 = datetime.datetime.now()
            print("Get_Data_Time: ", (end1 - start1))
            start2 = datetime.datetime.now()
            ##################################   FEEDING REAL DATA   #####################################
            train,loss,decoded_img,i_downsamp,args = SAE.add_train(batch_xs, batch_xs_last, batch_xs_next)
            ##################################   FEEDING REAL DATA   #####################################
            end2 = datetime.datetime.now()
            print("Get_Train_Time: ", (end2 - start2))
            # init_op = tf.initialize_all_tables()
            # init_var = tf.initialize_all_variables()
            # sess.run(init_op)
            # sess.run(init_var) 
            start3 = datetime.datetime.now()
            ###   Applying Optimizer   ###               
            train.run()
            ###   Applying Optimizer   ###
            end3 = datetime.datetime.now()
            print("Get_Apply_Grad_Time: ", (end3 - start3))
            #resized = tf.image.resize_images(decoded_img[0], [300, 300], method = 0)
            #resized = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
            #out_image = tf.image.encode_jpeg(resized, format="grayscale", quality = 100)
            #print(len(args))

            #print("----------------------------------------------------")
            #print(args[6].eval())
            #print("----------------------------------------------------")
            #print(args[7].eval())
            #print(args[0].eval().shape)
            #print(type(args[0].eval()))
            #print(args[0].eval().dtype)
            start4 = datetime.datetime.now()
            if ((epoch + 0) % 1 == 0):
                first_image_decoded = decoded_img[0]
                first_image_gray = i_downsamp[0]
                # print(type(first_image_decoded.eval()))
                # print(first_image_decoded.eval().shape)
                # print(first_image_decoded.eval().dtype)
                # print(type(first_image_gray.eval()))
                # print(first_image_gray.eval().shape)
                # print(first_image_gray.eval().dtype)
                image_decoded = first_image_decoded.eval()[:,:,0]
                image_gray = first_image_gray.eval()[:,:,0]
                # print(image_decoded.shape)
                # print(image_decoded.dtype)
                # print(image_gray.shape)
                # print(image_gray.dtype)
                print("Epoch: ", (epoch + 1), "image", i, "loss: ", loss.eval())
                plt.subplot(121); plt.imshow(image_gray, cmap = 'gray')
                plt.subplot(122); plt.imshow(image_decoded, cmap = 'gray')
                filename = '/media/best/A_Coding_Disk/sae_ws/output_image/' + str('%03d' % epoch) + str('%03d' % i) + '.jpg'
                # plt.show()
                plt.savefig(filename)
            end4 = datetime.datetime.now()
            print("Get_Image_Time: ", (end4 - start4))

                                                                                                
                