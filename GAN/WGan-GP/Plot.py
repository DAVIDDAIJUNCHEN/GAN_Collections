#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: plot images after training
@author: daijun.chen
"""

# import modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from WGanGPModel import WGanGP
import tensorflow.contrib.slim as slim
import imageio
from tensorflow.examples.tutorials.mnist import input_data

# disable the warning, does not enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prepare data needed to be inferred
mnist = input_data.read_data_sets("./data/", one_hot=False)

# define InfoGAN model with parameters
rand_dim = 40
input_dim = 784
lipschitz = 1.0
lambda_coef = 10.0

Model = WGanGP(rand_dim, input_dim)
_ = Model.build_model(rand_dim, input_dim)

# generate images by using pre-trained model and labels
batch_size = 100
width = 28
height = 28

num_check_row = 10
num_check_col = 10

images_origin, _ = mnist.test.next_batch(batch_size)

img_dir = './img_out/negative_loss_d/'

# make dirs to store images
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

load_dir = './logs/negative_loss_d/'
load_epochs = list(range(0, 10))# 20

# plot images
for load_epoch in load_epochs:
    gen_images = Model.generate(images_origin, load_dir, load_epoch)

    gen_images_reshape = gen_images.reshape((batch_size, width, height))
    check_images_reshape = gen_images_reshape[:(num_check_row*num_check_col)]

    imgs = np.ones((width*num_check_col + 5*num_check_col + 5,
                    height*num_check_row + 5*num_check_row + 5))

    for i in range((num_check_row-1)*(num_check_col-1)):
        row = np.int32(i/(num_check_col - 1))
        col = i % (num_check_row -1)
        imgs[(5 + 5*row + height*row):(5 + 5*row + height + height*row),
        (5 + 5*col + width*col):(5 + 5*col + width + width*col)] = check_images_reshape[i]

    imageio.imwrite(img_dir+'%s.png'%load_epoch, imgs)
