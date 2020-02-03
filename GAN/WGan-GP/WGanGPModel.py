#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: WGanGP
@author: daijun.chen
"""

# import modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from scipy import misc, ndimage

# create the WGanGP class
class WGanGP(object):
    """
    Build WGan-GP model class
    """
    # initialization function
    def __init__(self, rand_dim, input_dim):
        """
        Initialization the WGanGP class
        :param input_dim: dimension of real input sample
        :param rand_dim : dimension of random input in generator
        """
        self.rand_dim  = rand_dim
        self.input_dim = input_dim

    # generator function
    def generator(self, x):
        """
        Generator in GAN model
        :param x: random noise inputs
        :return:  generated fake samples
        """
        # compute reuse (True or False)
        reuse = len([v for v in tf.global_variables() if v.name.startswith('generator')]) > 0

        with tf.variable_scope('generator', reuse=reuse):
            x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, self.input_dim, activation_fn=tf.nn.sigmoid)

        return x

    # discriminator function
    def discriminator(self, x):
        """
        discriminate fake/true sample
        :param x: real/fake samples
        :return:  one dimension output
                  without activation function
        """
        # compute reuse (True/False)
        reuse = len([v for v in tf.global_variables() if v.name.startswith('discriminator')]) > 0

        with tf.variable_scope('discriminator', reuse=reuse):
            x = slim.fully_connected(x, num_outputs=128, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, num_outputs=32,  activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, num_outputs=1,   activation_fn=None)

        return x

    # build a GAN model with predefined generator and discriminator
    def build_model(self, rand_dim, input_dim):
        """
        build WGanGP model
        :return: generated fake samples
        """
        # define placeholder of discriminator input
        self.x = tf.placeholder(tf.float32, shape=[None, input_dim])

        # define batch size
        self.batch_size = tf.shape(self.x)[0]

        # define random input for generator
        z_rand = tf.random_normal((self.batch_size, rand_dim))

        # generate from [z_rand]
        self.gen = self.generator(z_rand)

        return self.gen

    # return generator and discriminator losses
    def loss(self, rand_dim, input_dim, lipschitz, lambda_coef):
        """
        compute generator and discriminator losses
        :param rand_dim:    dimension of random input in generator
        :param input_dim:   dimension of real input
        :param lipschitz:   lipschitz constant used in gradient penalty
        :param lambda_coef: coefficient of gradient penaty
        :return: generator and discriminator losses
        """
        # call build model function
        gen = self.build_model(rand_dim, input_dim)

        # random sample to get joint distribution
        eps = tf.random_uniform([self.batch_size, 1], minval=0.0, maxval=1.0)
        x_inter = eps * self.x + (1.0 - eps) * gen

        # compute the gradient at x_inter
        grad = tf.gradients(self.discriminator(x_inter), [x_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1))
        # MSE(grad_norm - lipschitz)
        #grad_penalty = tf.reduce_mean(tf.pow(grad_norm - lipschitz, 2)) * lambda_coef
        # mean(max(grad_norm - lipschitz, 0))
        grad_penalty = tf.reduce_mean(tf.nn.relu(grad_norm - lipschitz)) * lambda_coef
        # define loss function: discriminator loss and generator loss ???
        loss_d = tf.reduce_mean(self.discriminator(gen)) - tf.reduce_mean(self.discriminator(self.x))+\
                 grad_penalty
        loss_g = - tf.reduce_mean(self.discriminator(gen))

        return loss_g, loss_d

    def train(self, data_sets, batch_size, train_epochs, disp_step, disp_epoch, lr_disc, lr_gen,
              lipschitz, lambda_coef, save_dir='./log/', max_save_keep=2):
        """
        :param data_sets: data sets for training, data structure should be like mnist
        :param batch_size:  training batch size
        :param train_epochs: training epochs
        :param disp_step:   display loss values in each disp_step
        :param disp_epoch:  display accuracy values in each disp_epoch
        :param lr_disc:     learning rate in optimizing discriminator
        :param lr_gen:      learning rate in optimizing generator
        :param lipschitz:   lipschitz constant used in gradient penalty
        :param lambda_coef: regularizer coefficient on gradient penalty
        :param save_dir:    check point saving dir
        :param max_save_keep: max checkpoints can be kept in save_dir
        :return: ckpt model files saved in dir
        """
        # get the losses & discriminator accuracy before get trainable variables
        loss_g, loss_d = self.loss(self.rand_dim, self.input_dim, lipschitz, lambda_coef)

        # get the trainable variables
        train_vars = tf.trainable_variables()

        d_vars = [var for var in train_vars if 'discriminator' in var.name]
        g_vars = [var for var in train_vars if 'generator' in var.name]

        # define training steps
        disc_global_step = tf.Variable(0, trainable=False)
        gen_global_step = tf.Variable(0, trainable=False)

        # train discriminator
        train_disc = tf.train.AdamOptimizer(lr_disc).minimize(loss_d, var_list=d_vars, global_step=disc_global_step)
        # train generator
        train_gen = tf.train.AdamOptimizer(lr_gen).minimize(loss_g, var_list=g_vars, global_step=gen_global_step)

        # create saver and save dir
        self.saver = tf.train.Saver(max_to_keep=max_save_keep)

        # run training in session
        with tf.Session() as sess_train:
            # Initialize variables
            sess_train.run(tf.global_variables_initializer())

            # define the dev data sets for validation
            feeds_dev = {self.x: data_sets.validation.images}

            # define the test data sets for testing
            feeds_test = {self.x: data_sets.test.images}

            # start training epoches
            for epoch in range(train_epochs):
                num_batches = int(data_sets.train.num_examples / batch_size)
                # train in current epoch
                for step in range(num_batches):
                    batch_x, batch_y = data_sets.train.next_batch(batch_size)
                    feeds = {self.x: batch_x}
                    # run optimizer
                    l_d, _, l_d_step = sess_train.run([loss_d, train_disc, disc_global_step], feed_dict=feeds)
                    l_g, _, l_g_step = sess_train.run([loss_g, train_gen, gen_global_step], feed_dict=feeds)
                    # print disc_loss and gen_loss in each disp_step
                    if step % disp_step == 0:
                        print('Epoch:', '%2d' % (epoch + 1), 'Step:', '%3d' % (step + 1), 'disc_loss=',
                              "{:.4f}".format(l_d), ' ', 'gen_loss=', "{:.4f}".format(l_g))

                # print dev accuracy in each disp_epoch
                if epoch % disp_epoch == 0:
                    l_d_dev, l_g_dev = sess_train.run([loss_d, loss_g], feed_dict=feeds_dev)
                    print('\nLosses evaluated on Dev data set:')
                    print('Epoch:', '%2d' % (epoch + 1), 'disc_loss=', "{:.4f}".format(l_d_dev), ' ',
                          'gen_loss=', "{:.4f}".format(l_g_dev), '\n')

                # save model parameters in ckpts
                self.saver.save(sess_train, save_dir + "WGanGPModel.ckpt", global_step=epoch)
            # Show that model training is finished
            print('\nTraining is finished ! \n')

    def generate(self, images, save_dir, load_epoch='latest'):
        """
        generate images based on given labels
        :param images:     images used to compute batch size
        :param save_dir:   save dir for check points
        :param load_epoch: epoch you want to load
        :return: images generated randomly
        """
        feeds_generate = {self.x: images}
        saver = tf.train.Saver()

        # run generate image session
        with tf.Session() as sess_generate:
            sess_generate.run(tf.global_variables_initializer())
            # restore ckpt according to load_epoch
            if load_epoch == 'latest':
                ckpt = tf.train.latest_checkpoint(save_dir)
                try:
                    saver.restore(sess_generate, ckpt)
                except ValueError as ve:
                    print('There is no trained model')
                    print(ve)
            elif type(load_epoch) is int:
                try:
                    saver.restore(sess_generate, save_dir+'WGanGPModel.ckpt-'+str(load_epoch))
                except ValueError as ve:
                    print('load_epoch ', '%2d' % load_epoch, 'is larger than training epochs.')
            else:
                print('load_epoch should be an integer.')

            # generate images
            gen_images = sess_generate.run(self.gen, feed_dict=feeds_generate)

        return gen_images