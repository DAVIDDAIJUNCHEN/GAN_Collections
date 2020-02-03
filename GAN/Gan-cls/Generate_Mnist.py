#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: generate images for given labels and plot images for comparison
@author: daijun.chen
"""

# import modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from GanClsModel import GanCls
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

# disable the warning, does not enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prepare data needed to be inferred
mnist = input_data.read_data_sets("./data/", one_hot=False)

# define GanCls model with parameters
num_classes = 10
rand_dim = 38
con_dim = 2
input_dim = 784

Model = GanCls(num_classes, rand_dim, con_dim, input_dim)
_, _, _, _, _ = Model.loss(num_classes, rand_dim, con_dim, input_dim)

# generate images by using pre-trained model and labels
batch_size = 10
save_dir = './log/'
load_epoch = 'latest' # 20

images_origin, labels_generate = mnist.test.next_batch(batch_size)

# images_origin are only for batch_size in the model
gen_images = Model.generate(images_origin, labels_generate, save_dir, load_epoch)

# show generated and origin images
show_num = 10
width = 28
height = 28

f, a = plt.subplots(2, show_num, figsize=(10, 2))

for i in range(show_num):
    a[0][i].imshow(np.reshape(images_origin[i], (width, height)))
    a[1][i].imshow(np.reshape(gen_images[i],    (width, height)))

plt.draw()
plt.show()
