import os
import numpy as np
import tensorflow as tf
from nn.image_utils import get_image, save_images_to_one
import time
from nn.layers import *
from glob import glob
from utils.logger import logger, log
import tensorflow.contrib.learn as skflow
from base_model import BaseModel

# import matplotlib.pyplot as plt
# import seaborn as sns


class SteganalysisNet(BaseModel):
    def __init__(self, config, sess, stego_algorithm):
        super().__init__(sess, config, 'SteganalysisCLF')
        self.stego_algorithm = stego_algorithm

        self.images = tf.placeholder(tf.float32, [None] + list(self.conf.image_shape), name='images')

        self.data = self.get_images_names('train/*.%s' % self.conf.img_format)

    @log('Initializing neural network')
    def init_neural_network(self):
        self.stego_nn = self.network(self.stego_algorithm().encode(self.images))
        self.nn = self.network(self.images, reuse=True)

    @log('Initializing batch norms')
    def init_batch_norms(self):
        self.d_bn1 = batch_norm(self.conf.batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(self.conf.batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(self.conf.batch_size, name='d_bn3')

    @log('Initializing loss')
    def init_loss(self):
        self.nn_loss = binary_cross_entropy_with_logits(tf.zeros_like(self.nn), self.nn)
        self.nn_stego_loss = binary_cross_entropy_with_logits(tf.ones_like(self.stego_nn), self.stego_nn)
        self.loss = self.nn_loss + self.nn_stego_loss

    @log('Initializing variables')
    def init_vars(self):
        t_vars = tf.trainable_variables()

        self.nn_vars = [var for var in t_vars if 'nn_' in var.name]

    @log('Training')
    def train(self):
        if self.conf.need_to_load:
            self.load(self.conf.checkpoint_dir)

        data = glob(os.path.join(self.conf.data, "*.%s" % self.conf.img_format))
        logger.info('Total amount of images: %s' % len(data))
        # np.random.shuffle(data)

        optim = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        optim = optim.minimize(self.loss, var_list=self.nn_vars)

        tf.initialize_all_variables().run()

        sample_files = data[0:self.conf.sample_size]
        sample = [get_image(sample_file, self.conf.image_size, need_transform=True) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        batch_idxs = min(len(data), self.conf.train_size) / self.conf.batch_size

        logger.debug('Starting updating')
        for epoch in range(self.conf.epoch):
            losses = []

            logger.info('Starting epoch %s' % epoch)

            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]
                batch = [get_image(batch_file, self.conf.image_size, need_transform=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                self.sess.run(optim, feed_dict={self.images: batch_images})

                loss_stego = self.nn_stego_loss.eval({self.images: batch_images})
                loss_non_stego = self.nn_loss.eval({self.images: batch_images})

                losses.append(loss_stego + loss_non_stego)

                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f" %
                             (epoch, idx, batch_idxs, time.time() - start_time))
                logger.debug('[LOSS] Loss: %.8f' % (loss_stego + loss_non_stego))

                counter += 1
                if np.mod(counter, 100) == 0:
                    self.save(self.conf.checkpoint_dir, counter)

            stego_accuracy, non_stego_accuracy = self.accuracy()

            print('Epoch {:2d} error (stego, non_stego): {:3.1f}%, {:3.1f}%'.format(epoch + 1, 100 * stego_accuracy, 100 * non_stego_accuracy))

    @log('Accuracy')
    def accuracy(self):
        test_stego = [get_image(batch_file, self.conf.image_size, need_transform=True) for batch_file in self.get_images_names('test/stego_*.%s' % self.conf.img_format)]
        test_stego = np.array(test_stego).astype(np.float32)

        test_non_stego = [get_image(batch_file, self.conf.image_size, need_transform=True) for batch_file in self.get_images_names('test/empty_*.%s' % self.conf.img_format)]
        test_non_stego = np.array(test_non_stego).astype(np.float32)

        stego_answs = self.sess.run(self.nn, feed_dict={self.images: test_stego})
        non_stego_answs = self.sess.run(self.nn, feed_dict={self.images: test_non_stego})

        stego_mistakes = tf.not_equal(tf.ones_like(test_stego), tf.round(stego_answs))
        non_stego_mistakes = tf.not_equal(tf.zeros_like(test_non_stego), tf.round(non_stego_answs))

        return tf.reduce_mean(tf.cast(stego_mistakes, tf.float32)), tf.reduce_mean(tf.cast(non_stego_mistakes, tf.float32))

    @log('Plotting losses')
    def plot_loss(self):
        pass

    def network(self, img, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        df_dim = 64
        net = self.conv2d(img, df_dim, name='nn_h0_conv')
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 2, name='nn_fr_h1_conv')
        net = self.d_bn1(net)
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 4, name='nn_h2_conv')
        net = self.d_bn2(net)
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 8, name='nn_h3_conv')
        net = self.d_bn3(net)
        net = self.leaky_relu(net)

        out = self.linear(tf.reshape(net, [self.conf.batch_size, -1]), 1, 'nn_h3_lin')

        return tf.nn.sigmoid(out)
