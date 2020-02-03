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

# plot images corresponding to containing information
start_con = 0.00001
end_con = 0.99999
num_con = 10
width = 28
height = 28

plot_con = tf.placeholder(tf.float32, [batch_size, 2])
plot_rand = tf.random_normal((batch_size, rand_dim))
plot_z = tf.concat(axis=1, values=[tf.one_hot(labels_generate, depth=num_classes),
                                   plot_con, plot_rand])

plot_gen = Model.generator(plot_z)
plot_genout = tf.squeeze(plot_gen, -1)

plot_con1 = np.ones([batch_size, 2])
a = np.linspace(start_con, end_con, num_con)
b = np.linspace(start_con, end_con, batch_size)
y_input = np.ones([num_con])
figure = np.zeros((batch_size * width, batch_size * height))

saver = tf.train.Saver()
save_dir = './log/'
load_epoch = 'latest' # 20

with tf.Session() as sess_plot:
    sess_plot.run(tf.global_variables_initializer())
    if load_epoch == 'latest':
        ckpt = tf.train.latest_checkpoint(save_dir)
        try:
            saver.restore(sess_plot, ckpt)
        except ValueError as ve:
            print('There is no trained model')
            print(ve)
    elif type(load_epoch) is int:
        try:
            saver.restore(sess_plot, save_dir+'GanClsModel.ckpt-'+str(load_epoch))
        except ValueError as ve:
            print('load_epoch ', '%2d' % load_epoch, 'is larger than training epochs.')
    else:
        print('load_epoch should be an integer.')

    for i in range(num_con):
        for j in range(batch_size):
            plot_con1[j][0] = a[i]
            plot_con1[j][1] = b[j]
            y_input[j] = j % 10  # take the mod ==> [0, 9]

        feeds = {plot_con: plot_con1}
        plot_genout_run = sess_plot.run(plot_genout, feed_dict=feeds)

        for jj in range(batch_size):
            digit = plot_genout_run[jj].reshape(width, height)
            figure[i * width: (i + 1) * width, jj * height: (jj + 1) * height] = digit

    plt.figure(figsize=(width, height))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
