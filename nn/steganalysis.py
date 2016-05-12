import os
import numpy as np
import tensorflow as tf
from nn.image_utils import get_image, save_images_to_one
import time
from nn.layers import *
from glob import glob
from utils.logger import logger, log
import tensorflow.contrib.learn as skflow
from nn.base_model import BaseModel
import functools


# import matplotlib.pyplot as plt
# import seaborn as sns


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class SteganalysisNet(BaseModel):
    def __init__(self, config, sess, stego_algorithm, image_shape=(64, 64, 3)):
        super().__init__(sess, config, 'SteganalysisCLF')
        self.stego_algorithm = stego_algorithm
        self.image_shape = image_shape

        self.images = tf.placeholder(tf.float32, [self.conf.batch_size] + list(self.image_shape))
        self.target = tf.placeholder(tf.float32, [self.conf.batch_size, ])

        self.data = self.get_images_names('train/*.%s' % self.conf.img_format)
        self.init_batch_norms()

        # init
        self.loss
        self.optimize
        self.network

    @log('Initializing batch norms')
    def init_batch_norms(self):
        self.d_bn1 = batch_norm(self.conf.batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(self.conf.batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(self.conf.batch_size, name='d_bn3')

    @log('Get targets for given files')
    def get_targets(self, batch_files):
        get_tar = lambda x: 'stego_' in os.path.split(x)[-1]
        return np.array([get_tar(f) for f in batch_files], dtype=np.float32)

    @log('Training')
    def train(self):
        if self.conf.need_to_load:
            self.load(self.conf.checkpoint_dir)

        data = self.data
        logger.info('Total amount of images: %s' % len(data))
        # np.random.shuffle(data)

        tf.initialize_all_variables().run()

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

                batch_targets = self.get_targets(batch_files)

                self.sess.run(self.optimize, feed_dict={self.images: batch_images, self.target: batch_targets})
                loss = self.loss.eval({self.images: batch_images, self.target: batch_targets})

                losses.append(loss)

                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f" %
                             (epoch, idx, batch_idxs, time.time() - start_time))
                logger.debug('[LOSS] Loss: %.8f' % loss)

                counter += 1
                if np.mod(counter, 600) == 0:
                    # self.save(self.conf.checkpoint_dir, counter)

                    stego_accuracy = self.accuracy()

                    print('Epoch {:2d} error: {:3.1f}%'.format(epoch + 1, 100 * stego_accuracy))

    def get_accuracy(self, test_stego, test_non_stego):
        stego_answs = self.sess.run(self.network, feed_dict={self.images: test_stego})
        non_stego_answs = self.sess.run(self.network, feed_dict={self.images: test_non_stego})

        stego_mistakes = tf.not_equal(tf.ones_like(test_stego), tf.round(stego_answs))
        non_stego_mistakes = tf.not_equal(tf.zeros_like(test_non_stego), tf.round(non_stego_answs))

        return (tf.reduce_mean(tf.cast(stego_mistakes, tf.float32)).eval() +
                tf.reduce_mean(tf.cast(non_stego_mistakes, tf.float32)).eval()) / 2

    @log('Accuracy')
    def accuracy(self):
        test_stego = [get_image(batch_file, self.conf.image_size, need_transform=True) for batch_file in
                      self.get_images_names('test/stego_*.%s' % self.conf.img_format)]
        test_stego = np.array(test_stego).astype(np.float32)

        test_non_stego = [get_image(batch_file, self.conf.image_size, need_transform=True) for batch_file in
                          self.get_images_names('test/empty_*.%s' % self.conf.img_format)]
        test_non_stego = np.array(test_non_stego).astype(np.float32)

        accuracies = []

        batch_idxs = min(len(test_stego), self.conf.train_size) / self.conf.batch_size

        logger.debug('Starting iteration')
        for idx in range(0, int(batch_idxs)):
            batch_files_stego = test_stego[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]

            batch_files_non_stego = test_non_stego[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]

            accuracies.append(self.get_accuracy(batch_files_stego, batch_files_non_stego))

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return np.mean(accuracies)

    @log('Plotting losses')
    def plot_loss(self):
        pass

    @lazy_property
    def saver(self):
        return tf.train.Saver()

    @lazy_property
    def loss(self):
        return binary_cross_entropy_with_logits(self.target, self.network)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.network, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1).minimize(self.loss)

    @lazy_property
    def network(self):
        df_dim = 64
        net = self.conv2d(self.images, df_dim, name='nn_h0_conv')
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 2, name='nn_h1_conv')
        net = self.d_bn1(net)
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 4, name='nn_h2_conv')
        net = self.d_bn2(net)
        net = self.leaky_relu(net)

        net = self.conv2d(net, df_dim * 8, name='nn_h3_conv')
        net = self.d_bn3(net)
        net = self.leaky_relu(net)

        net = tf.reshape(net, [self.conf.batch_size, -1])
        out = self.linear(net, 1, 'nn_h4_lin')

        return tf.nn.sigmoid(out)
