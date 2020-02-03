#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: Train AeGan model on mnist data
@author: daijun.chen
"""

# import modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from AeGanModel import AeGAN
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

# disable the warning, does not enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input data
mnist = input_data.read_data_sets("./data/", one_hot=False)

# define InfoGAN model with parameters
num_classes = 10
rand_dim = 38
con_dim = 2
input_dim = 784

Model = AeGAN(num_classes, rand_dim, con_dim, input_dim)

# define hyper-parameters and train model
batch_size = 10
trainGAN_epochs = 20
trainAE_epochs = 20
display_epoch = 1
display_step = 100
lr_discriminator = 0.0001
lr_generator = 0.001
lr_inverse_generator = 0.01

Model.train(mnist, batch_size, trainGAN_epochs, trainAE_epochs, display_step, display_epoch,
            lr_discriminator, lr_generator, lr_inverse_generator)
