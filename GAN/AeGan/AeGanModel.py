#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 24 21:16:30 2020
Module: AeGanModel
@author: daijun.chen
"""

# import modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import sys
from scipy.stats import norm
sys.path.append(os.path.realpath('..'))  # add upper dir into path
from InfoGan.InfoGanModel import InfoGAN


# create the AeGanModel class from InfoGanModel class
class AeGAN(InfoGAN):
    """
    Build AeGAN class by inheriting from InfoGAN model class
    """
    def inverse_generator(self, x):
        """
        encode the real samples to a lower feature space
        :param x: real samples
        :return:  lower dimension representation
        """
        leaky_relu = self.leaky_relu
        reuse = len([t for t in tf.global_variables() if t.name.startswith('inverse_generator')]) > 0
        with tf.variable_scope('inverse_generator', reuse=reuse):     # inverse_generator variable scope
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            x = slim.conv2d(x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
            x = slim.conv2d(x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)

            x = slim.flatten(x)

            shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
            inv_gen = slim.fully_connected(shared_tensor, num_outputs=50, activation_fn=leaky_relu)

        return inv_gen

    def build_AeModel(self, classes_dim, rand_dim, con_dim, input_dim):
        """
        generate fake auto-encoder sample (fake cycle) and
        generate real auto-encoder sample (real cycle)
        :param classes_dim: number of classes
        :param rand_dim:    dimension of random component
        :param con_dim:     dimension of component containing information
        :param input_dim:   dimension of real inputs
        :return:  auto-encoder generated samples
        """
        # build an InfoGan model
        disc_real_fake, recog_cat_real_fake, recog_cont_real_fake = self.build_model(classes_dim, rand_dim,
                                                                                     con_dim, input_dim)
        # fake cycle: random noise ==> fake sample ==> low-dim feature ==> fake sample
        gen_fake = self.gen
        inv_gen_fake = self.inverse_generator(gen_fake)
        ae_gen_fake = self.generator(inv_gen_fake)

        # real cycle: real sample ==> low-dim feature ==> fake sample (by generator)
        inv_gen_real = self.inverse_generator(self.x)
        ae_gen_real = self.generator(inv_gen_real)

        ae_gen = {'real': ae_gen_real, 'fake': ae_gen_fake}

        return disc_real_fake, recog_cat_real_fake, recog_cont_real_fake, ae_gen

    def loss_ae(self, classes_dim, rand_dim, con_dim, input_dim):
        """
        compute the auto-encoder loss
        :param classes_dim: number of classes
        :param rand_dim:    dimension of random component
        :param con_dim:     dimension of component containing information
        :param input_dim:   dimension of real inputs
        :return:  auto-encoder loss
        """
        disc, self.recog_cat, recog_cont, ae_gen = self.build_AeModel(classes_dim, rand_dim, con_dim, input_dim)
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
        loss_cr = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.recog_cat['real'], labels=self.y))
        loss_cf = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.recog_cat['fake'], labels=self.y))
        loss_c = (loss_cr + loss_cf) / 2.0

        # hidden information loss
        loss_con = tf.reduce_mean(tf.square(recog_cont['fake'] - self.z_con))

        # define the discriminator accuracy on fake and real
        Acc_d_real = tf.reduce_mean(tf.cast(tf.sigmoid(disc['real']) > 0.5, tf.float32))
        Acc_d_fake = tf.reduce_mean(tf.cast(tf.sigmoid(disc['fake']) < 0.5, tf.float32))
        Acc_d_total = Acc_d_real * 0.5 + Acc_d_fake * 0.5

        Acc_d = {'real': Acc_d_real, 'fake': Acc_d_fake, 'total': Acc_d_total}

        # ? why only try ||ae_gen['fake'] - gen||, but not try ||ae_gen['real'] - x|| ?
        loss_ae = tf.reduce_mean(tf.pow(ae_gen['fake'] - self.gen, 2))

        return loss_d, loss_g, loss_c, loss_con, loss_ae, Acc_d

    def train(self, data_sets, batch_size, trainGAN_epochs, trainAE_epochs, disp_step, disp_epoch, lr_disc, lr_gen, lr_ae,
              save_dir='./log/', max_save_keep=2):
        """
        :param data_sets: data sets for training, data structure should be like mnist
        :param batch_size: training batch size
        :param trainGAN_epochs: GAN training epochs
        :param trainAE_epochs:  AE training epochs
        :param disp_step:  display loss values in each disp_step
        :param disp_epoch: display accuracy values in each disp_epoch
        :param lr_disc:    learning rate in optimizing discriminator
        :param lr_gen:     learning rate in optimizing generator
        :param lr_ae:      learning rate in optimizing auto-encoder (inverse-generator)
        :param save_dir:   check point saving dir
        :param max_save_keep: max checkpoints can be kept in save_dir
        :return:           ckpt model files saved in dir
        """
        # get the losses & discriminator accuracy before get trainable variables
        loss_d, loss_g, loss_c, loss_con, loss_ae, Acc_d = self.loss_ae(self.classes_dim, self.rand_dim,
                                                                        self.con_dim, self.input_dim)
        # get the trainable variables
        train_vars = tf.trainable_variables()

        d_vars = [var for var in train_vars if 'discriminator' in var.name]
        g_vars = [var for var in train_vars if 'generator' in var.name]
        ae_vars = [var for var in train_vars if 'inverse_generator' in var.name]

        # define training steps
        disc_global_step = tf.Variable(0, trainable=False)
        gen_global_step = tf.Variable(0, trainable=False)
        ae_global_step = tf.Variable(0, trainable=False)  # tf.contrib.framework.get_or_create_global_step()

        # train discriminator
        train_disc = tf.train.AdamOptimizer(lr_disc).minimize(loss_d + loss_c + loss_con, var_list=d_vars,
                                                              global_step=disc_global_step)
        # train generator
        train_gen = tf.train.AdamOptimizer(lr_gen).minimize(loss_g + loss_c + loss_con, var_list=g_vars,
                                                            global_step=gen_global_step)
        # train auto-encoder / inverse generator
        train_ae = tf.train.AdamOptimizer(lr_ae).minimize(loss_ae, var_list=ae_vars,
                                                          global_step=ae_global_step)

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

            num_batches = int(data_sets.train.num_examples / batch_size)
            # start GAN model training epochs
            for epoch in range(trainGAN_epochs):
                # train GAN in current epoch
                for step in range(num_batches):
                    batch_x, batch_y = data_sets.train.next_batch(batch_size)
                    feeds = {self.x: batch_x, self.y: batch_y}
                    # run optimizer
                    l_d, _, l_d_step = sess_train.run([loss_d, train_disc, disc_global_step], feed_dict=feeds)
                    l_g, _, l_g_step = sess_train.run([loss_g, train_gen, gen_global_step], feed_dict=feeds)
                    # print disc_loss and gen_loss in each disp_step
                    if step % disp_step == 0:
                        print('Epoch:', '%2d' % (epoch + 1), 'Step:', '%3d' % (step + 1), 'disc_loss=',
                              "{:.4f}".format(l_d), ' ', 'gen_loss=', "{:.4f}".format(l_g))

                # print dev accuracy in each disp_epoch
                if epoch % disp_epoch == 0:
                    accuracy_d = sess_train.run(Acc_d, feed_dict=feeds_dev)
                    print('\nAccuracy evaluated on Dev data set:')
                    print('Epoch:', '%2d' % (epoch + 1), 'acc_d_real=', "{:.9f}".format(accuracy_d['real']), ' ',
                          'acc_d_fake=', "{:.9f}".format(accuracy_d['fake']), ' ', 'acc_d_total=',
                          "{:.9f}".format(accuracy_d['total']), '\n')

                # save model parameters in ckpts
                self.saver.save(sess_train, save_dir + "InfoGanModel.ckpt", global_step=epoch)
            # Show that model training is finished
            print('\nInfoGAN training is finished ! \n')

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


            # start auto-encoder model training
            for epoch in range(trainAE_epochs):
                # train AE in current epoch
                for step in range(num_batches):
                    batch_x, batch_y = data_sets.train.next_batch(batch_size)
                    feeds = {self.x: batch_x, self.y: batch_y}
                    # run optimizer
                    l_ae, _, l_ae_step = sess_train.run([loss_ae, train_ae, ae_global_step], feed_dict=feeds)
                    # print ae_loss in each disp_step
                    if step % disp_step == 0:
                        print('Epoch:', '%2d' % (epoch+1), 'Step:', '%3d' % (step+1), 'ae_loss=',
                              "{:.4f}".format(l_ae), ' ', 'ae_loss=', "{:.3f}".format(l_ae))

            # print dev loss in each disp_epoch
                if epoch % disp_epoch == 0:
                    l_ae_dev = sess_train.run(loss_ae, feed_dict=feeds_dev)
                    print('\nAE loss evaluated on Dev data set:')
                    print('Epoch:', '%2d' % (epoch + 1), 'ae_loss=', "{:.9f}".format(l_ae_dev), '\n')

                # save model parameters in ckpts
                self.saver.save(sess_train, save_dir+"AeGanModel.ckpt", global_step=epoch)
            # Show that model training is finished
            print('\nInverse-Generator training is finished ! \n')
