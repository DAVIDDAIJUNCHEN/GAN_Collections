#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: InfoGanModel
@author: daijun.chen
"""

# import modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim

# Class: InfoGAN MODEL
class InfoGAN(object):
    def __init__(self, classes_dim, rand_dim, con_dim, input_dim):
        """
        Initialize the model hyperparameters
        classes_dim: number of classes
        rand_dim: dimension of random input component
        con_dim: dimension of component containing information
        input_dim: dimension of real input sample
        """
        self.classes_dim = classes_dim
        self.rand_dim = rand_dim
        self.con_dim = con_dim
        self.input_dim = input_dim
    
    def leaky_relu(self, x, slope=0.01):
        """
        leaky relu activation function
        """
        return tf.where(tf.greater(x, 0), x, x*slope)
    
    def generator(self, x):
        """
        generate fake samples from noise vector
        x: random noise vector
        """
        reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
        with tf.variable_scope('generator', reuse=reuse):        # generator name scope
            x = slim.fully_connected(x, 1024)
            x = slim.batch_norm(x, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, 7*7*128)
            x = slim.batch_norm(x, activation_fn=tf.nn.relu)
            x = tf.reshape(x, [-1, 7, 7, 128])
            
            x = slim.conv2d_transpose(x, 64, kernel_size=[4,4], stride=2, activation_fn=None)
            
            x = slim.batch_norm(x, activation_fn=tf.nn.relu)
            
            z = slim.conv2d_transpose(x, 1, kernel_size=[4,4], stride=2, activation_fn=tf.nn.sigmoid)
        
        return z
            
    def discriminator(self, x, num_classes, num_cont): # ?
        """
        discriminate fake/true samples + category + 
        """
        leaky_relu = self.leaky_relu
        reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
        with tf.variable_scope('discriminator', reuse=reuse):    # discriminator name scope
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            x = slim.conv2d(x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
            x = slim.conv2d(x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
            
            x = slim.flatten(x)
            
            shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
            recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn=leaky_relu)
            
            # discrimination tensor, 0/1
            disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
            disc = tf.squeeze(disc, -1)
            
            # recognize category
            recog_cat = slim.fully_connected(recog_shared, num_outputs=num_classes, activation_fn=None)
            
            # recognize hidden information
            recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)
        
        return disc, recog_cat, recog_cont
    
    def build_model(self, classes_dim, rand_dim, con_dim, input_dim):
        """
        build InfoGAN model 
        """
        # define placeholder of discriminator input
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        # define placeholder of generator classes info
        self.y = tf.placeholder(tf.int32, [None])

        # define batch_size = tf.shape(self.x)[0]
        self.batch_size = tf.shape(self.x)[0]

        # define z = [y, z_con, z_rand]
        self.z_con = tf.random_normal((self.batch_size, con_dim))
        z_rand = tf.random_normal((self.batch_size, rand_dim))
        z = tf.concat(axis=1, values=[tf.one_hot(self.y, depth=classes_dim), self.z_con, z_rand])
        
        # generate from [y, z_con, z_rand] 
        self.gen = self.generator(z)
        self.gen_out = tf.squeeze(self.gen, -1)
        
        # discriminate real / fake samples
        disc_real, recog_cat_real, recog_cont_real = self.discriminator(self.x, classes_dim, con_dim)
        disc_fake, recog_cat_fake, recog_cont_fake = self.discriminator(self.gen, classes_dim, con_dim)
        # pred_class_fake = tf.argmax(recog_cat_fake, dimension=1)

        
        disc_real_fake = {'real':disc_real, 'fake':disc_fake}
        recog_cat_real_fake = {'real':recog_cat_real, 'fake':recog_cat_fake}
        recog_cont_real_fake = {'real':recog_cont_real, 'fake':recog_cont_fake}
        
        return disc_real_fake, recog_cat_real_fake, recog_cont_real_fake 
    
    # loss function: loss_d, loss_g, loss_c, loss_con
    def loss(self, classes_dim, rand_dim, con_dim, input_dim):
        """
        :param classes_dim: number of classes
        :param rand_dim:    dimension of random component
        :param con_dim:     dimension of component containing information
        :param input_dim:   dimension of real inputs
        :return: loss_d, loss_g, loss_c, loss_con, Acc_d (accuracy of discriminator)
        """
        # build model
        disc, self.recog_cat, recog_cont = self.build_model(classes_dim, rand_dim, con_dim, input_dim)
        
        # define labels
        y_real = tf.ones([self.batch_size])
        y_fake = tf.zeros([self.batch_size])
        
        # define discrimination loss
        loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc['real'],
                                                                          labels=y_real))
        loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc['fake'],
                                                                          labels=y_fake))
        loss_d = (loss_d_r + loss_d_f) / 2.0
        
        # define generator loss
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc['fake'], labels=y_real))
        
        # define factor loss
        loss_cr = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.recog_cat['real'], labels=self.y))
        loss_cf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.recog_cat['fake'], labels=self.y))
        loss_c = (loss_cr + loss_cf) / 2.0
        
        # hidden information loss
        loss_con = tf.reduce_mean(tf.square(recog_cont['fake'] - self.z_con))

        # define the discriminator accuracy on fake and real
        Acc_d_real = tf.reduce_mean(tf.cast(tf.sigmoid(disc['real']) > 0.5, tf.float32))
        Acc_d_fake = tf.reduce_mean(tf.cast(tf.sigmoid(disc['fake']) < 0.5, tf.float32))
        Acc_d_total = Acc_d_real * 0.5 + Acc_d_fake*0.5

        Acc_d = {'real': Acc_d_real, 'fake': Acc_d_fake, 'total': Acc_d_total}

        return loss_d, loss_g, loss_c, loss_con, Acc_d
    
    def train(self, data_sets, batch_size, train_epochs, disp_step, disp_epoch, lr_disc, lr_gen,
              save_dir='./log/', max_save_keep=2):
        """
        :param data_sets: data sets for training, data structure should be like mnist
        :param batch_size: training batch size
        :param train_epochs: training epochs
        :param disp_step:  display loss values in each disp_step
        :param disp_epoch: display accuracy values in each disp_epoch
        :param lr_disc:    learning rate in optimizing discriminator
        :param lr_gen:     learning rate in optimizing generator
        :param save_dir:   check point saving dir
        :param max_save_keep: max checkpoints can be kept in save_dir
        :return: ckpt model files saved in dir
        """
        # get the losses & discriminator accuracy before get trainable variables
        loss_d, loss_g, loss_c, loss_con, Acc_d = self.loss(self.classes_dim, self.rand_dim,
                                                            self.con_dim, self.input_dim)
        # get the trainable variables
        train_vars = tf.trainable_variables()

        d_vars = [var for var in train_vars if 'discriminator' in var.name]
        g_vars = [var for var in train_vars if 'generator' in var.name]
        
        # define training steps 
        disc_global_step = tf.Variable(0, trainable=False)
        gen_global_step = tf.Variable(0, trainable=False)

        # train discriminator
        train_disc = tf.train.AdamOptimizer(lr_disc).minimize(loss_d + loss_c + loss_con, var_list=d_vars,
                                                              global_step=disc_global_step)
        # train generator        
        train_gen = tf.train.AdamOptimizer(lr_gen).minimize(loss_g + loss_c + loss_con, var_list=g_vars,
                                                              global_step=gen_global_step)

        # create saver and save dir
        self.saver = tf.train.Saver(max_to_keep=max_save_keep)

        # run training in session
        with tf.Session() as sess_train:
            # Initialize variables
            sess_train.run(tf.global_variables_initializer())

            # define the dev data sets for validation
            feeds_dev = {self.x: data_sets.validation.images,
                        self.y: data_sets.validation.labels}

            # define the test data sets for testing
            feeds_test = {self.x: data_sets.test.images,
                        self.y: data_sets.test.labels}

            # start training epoches
            for epoch in range(train_epochs):
                num_batches = int(data_sets.train.num_examples/batch_size)
                # train in current epoch
                for step in range(num_batches):
                    batch_x, batch_y = data_sets.train.next_batch(batch_size)
                    feeds = {self.x: batch_x, self.y: batch_y}
                    # run optimizer 
                    l_d, _, l_d_step = sess_train.run([loss_d, train_disc, disc_global_step], feed_dict=feeds)
                    l_g, _, l_g_step = sess_train.run([loss_g, train_gen, gen_global_step], feed_dict=feeds)
                    # print disc_loss and gen_loss in each disp_step
                    if step % disp_step == 0:
                        print('Epoch:', '%2d' % (epoch+1), 'Step:', '%3d' % (step+1), 'disc_loss=',
                              "{:.4f}".format(l_d), ' ', 'gen_loss=', "{:.4f}".format(l_g))

                # print dev accuracy in each disp_epoch
                if epoch % disp_epoch == 0:
                    accuracy_d = sess_train.run(Acc_d, feed_dict=feeds_dev)
                    print('\nAccuracy evaluated on Dev data set:')
                    print('Epoch:', '%2d' % (epoch + 1), 'acc_d_real=', "{:.9f}".format(accuracy_d['real']), ' ',
                        'acc_d_fake=', "{:.9f}".format(accuracy_d['fake']), ' ', 'acc_d_total=',
                        "{:.9f}".format(accuracy_d['total']), '\n')

                # save model parameters in ckpts
                self.saver.save(sess_train, save_dir+"InfoGanModel.ckpt", global_step=epoch)
            # Show that model training is finished
            print('\nTraining is finished ! \n')

            # print dev accuracy after training is finished
            accuracy_d_dev = sess_train.run(Acc_d, feed_dict=feeds_dev)
            print('\nFinal accuracy evaluated on Dev data set:')
            print('acc_d_real=', "{:.9f}".format(accuracy_d_dev['real']), ' ',
                  'acc_d_fake=', "{:.9f}".format(accuracy_d_dev['fake']), ' ',
                  'acc_d_total=', "{:.9f}".format(accuracy_d_dev['total']))

            # print test accuracy after training is finished
            accuracy_d_test = sess_train.run(Acc_d, feed_dict=feeds_test)
            print('\nFinal accuracy evaluated on Test data set:')
            print('acc_d_real=', "{:.9f}".format(accuracy_d_test['real']), ' ',
                  'acc_d_fake=', "{:.9f}".format(accuracy_d_test['fake']), ' ',
                  'acc_d_total=', "{:.9f}".format(accuracy_d_test['total']))

    def inference(self, images, save_dir, load_epoch='latest'):
        """
        Inference the labels of input images (Classification)
        :return: labels of input images
        """
        feeds_inference = {self.x: images}

        saver = tf.train.Saver()

        # run inference session
        with tf.Session() as sess_inference:
            sess_inference.run(tf.global_variables_initializer())
            # restore ckpt according to load_epoch
            if load_epoch == 'latest':
                ckpt = tf.train.latest_checkpoint(save_dir)
                try:
                    saver.restore(sess_inference, ckpt)
                except ValueError as ve:
                    print('There is no trained model')
                    print(ve)
            elif type(load_epoch) is int:
                try:
                    saver.restore(sess_inference, save_dir+'InfoGanModel.ckpt-'+str(load_epoch))
                except ValueError as ve:
                    print('load_epoch ', '%2d' % load_epoch, 'is larger than training epochs.')
            else:
                print('load_epoch should be an integer.')

            # inference (classification)
            pred_class_real = sess_inference.run(self.recog_cat['real'], feed_dict=feeds_inference)
            print("The predicted labels are", np.argmax(pred_class_real, axis=1))

    def generate(self, images, labels, save_dir, load_epoch='latest'):
        """
        generate images based on given labels
        :param labels:     labels (one-hot=False)
        :param save_dir:   save dir for check points
        :param load_epoch: epoch you want to load
        :return:           images corresponding to labels
        """
        feeds_generate = {self.x: images, self.y: labels}
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
                    saver.restore(sess_generate, save_dir+'InfoGanModel.ckpt-'+str(load_epoch))
                except ValueError as ve:
                    print('load_epoch ', '%2d' % load_epoch, 'is larger than training epochs.')
            else:
                print('load_epoch should be an integer.')

            # generate images
            gen_images = sess_generate.run(self.gen, feed_dict=feeds_generate)

        return gen_images
