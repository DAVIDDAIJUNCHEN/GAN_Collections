#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: Train WGan-GP model on mnist data
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
from tensorflow.examples.tutorials.mnist import input_data

# disable the warning, does not enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input data
mnist = input_data.read_data_sets("./data/", one_hot=False)

# define InfoGAN model with parameters
rand_dim = 40
input_dim = 784

Model = WGanGP(rand_dim, input_dim)

# define hyper-parameters and train model
batch_size = 10
train_epochs = 10
display_epoch = 1
display_step = 100
lr_discriminator = 0.0001
lr_generator = 0.001
lipschitz = 1.0
lambda_coef = 10.0

# save dirs + max_save_keep
save_dir = './logs/negative_loss_d/'
max_save_keep = 10

if not os.path.exists('./logs/negative_loss_d/'):
    os.makedirs('./logs/negative_loss_d/')

Model.train(mnist, batch_size, train_epochs, display_step, display_epoch,
            lr_discriminator, lr_generator, lipschitz, lambda_coef,
            save_dir=save_dir, max_save_keep=max_save_keep)
