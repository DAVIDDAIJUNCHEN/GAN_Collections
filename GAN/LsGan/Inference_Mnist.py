#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: inference (classify) real images
@author: daijun.chen
"""

# import modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from LSGanModel import LSGan
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

# disable the warning, does not enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prepare data needed to be inferred
mnist = input_data.read_data_sets("./data/", one_hot=False)

# define LSGan model with parameters
num_classes = 10
rand_dim = 38
con_dim = 2
input_dim = 784

Model = LSGan(num_classes, rand_dim, con_dim, input_dim)
_, _, _, _, _ = Model.loss(num_classes, rand_dim, con_dim, input_dim)

# classify images by using pre-trained model
batch_size = 10
save_dir = './log/'
load_epoch = 'latest' # 20

images_inference, labels_inference = mnist.test.next_batch(batch_size)

Model.inference(images_inference, save_dir, load_epoch)

print("True labels are", labels_inference)
