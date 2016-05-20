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
        super().__init__(sess, config)
        self.stego_algorithm = stego_algorithm
        self.image_shape = image_shape

        self.images = tf.placeholder(tf.float32, [self.conf.batch_size] + list(self.image_shape))
        self.target = tf.placeholder(tf.float32, [self.conf.batch_size, ])

        self.data = self.get_images_names('train/*.%s' % self.conf.img_format)

        # init
        self.loss
        self.optimize
        self.network

    def image_processing_layer(self, X):
        K = 1 / 12. * tf.constant([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=tf.float32)

        # kernel = tf.zeros([5, 5, self.conf.image_size[-1], self.conf.image_size[-1]])

        kernel = tf.pack([K, K, K])
        kernel = tf.pack([kernel, kernel, kernel])

        return tf.nn.conv2d(X, tf.transpose(kernel, [2, 3, 0, 1]), [1, 1, 1, 1], padding='SAME')

    @staticmethod
    def get_targets(batch_files):
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

        stego_accuracy = self.accuracy()

        accuracies = []
        accuracies_steps = []

        logger.debug('Starting updating')
        for epoch in range(self.conf.epoch):
            losses = []

            logger.info('Starting epoch %s' % epoch)

            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]
                batch = [get_image(batch_file, self.conf.image_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_targets = self.get_targets(batch_files)

                self.sess.run(self.optimize, feed_dict={self.images: batch_images, self.target: batch_targets})
                loss = self.loss.eval({self.images: batch_images, self.target: batch_targets})

                losses.append(loss)

                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f, Loss: %8f, accuracy: %8f" %
                             (epoch, idx, batch_idxs, time.time() - start_time, loss, stego_accuracy))

                counter += 1

            # SAVE after each epoch
            self.save(self.conf.checkpoint_dir, counter)

            stego_accuracy = self.accuracy(n_files=-1)

            accuracies.append(stego_accuracy)
            accuracies_steps.append(counter)

            max_acc_idx = np.argmax(accuracies)
            logger.info('[TEST] Epoch {:2d} error: {:3.1f}%'.format(epoch + 1, 100 * stego_accuracy))
            logger.info('[TEST] Max accuracy: %s, step: %s, epoch: %s' % (accuracies[max_acc_idx],
                                                                          accuracies_steps[max_acc_idx], epoch))

    def get_accuracy(self, test_stego, test_non_stego):
        stego_answs = self.sess.run(self.network, feed_dict={self.images: test_stego})
        non_stego_answs = self.sess.run(self.network, feed_dict={self.images: test_non_stego})

        # print(tf.ones((1, test_stego.shape[0])).eval(), tf.reshape(tf.round(stego_answs), (1, -1)).eval())
        stego_mistakes = tf.equal(tf.ones((1, test_stego.shape[0])), tf.reshape(tf.round(stego_answs), (1, -1)))
        non_stego_mistakes = tf.equal(tf.zeros((1, test_non_stego.shape[0])), tf.reshape(tf.round(non_stego_answs), (1, -1)))

        return (tf.reduce_mean(tf.cast(stego_mistakes, tf.float32)).eval() +
                tf.reduce_mean(tf.cast(non_stego_mistakes, tf.float32)).eval()) / 2

    def accuracy(self, test_dir='test', n_files=2**12):
        # TODO use get_labels
        logger.info('[TEST], test data folder: %s, n_files: %s' % (test_dir, 2 * n_files))
        test_stego = [get_image(batch_file, self.conf.image_size) for batch_file in
                      self.get_images_names('%s/stego_*.%s' % (test_dir, self.conf.img_format))[:n_files]]

        test_stego = np.array(test_stego).astype(np.float32)

        test_non_stego = [get_image(batch_file, self.conf.image_size) for batch_file in
                          self.get_images_names('%s/empty_*.%s' % (test_dir, self.conf.img_format))[:n_files]]
        test_non_stego = np.array(test_non_stego).astype(np.float32)

        accuracies = []

        batch_idxs = min(len(test_stego), self.conf.train_size) / self.conf.batch_size

        # logger.debug('Starting iteration')
        for idx in range(0, int(batch_idxs)):
            batch_files_stego = test_stego[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]

            batch_files_non_stego = test_non_stego[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]

            accuracies.append(self.get_accuracy(batch_files_stego, batch_files_non_stego))

        return np.mean(accuracies)

    @log('Plotting losses')
    def plot_loss(self):
        pass

    @lazy_property
    def saver(self):
        return tf.train.Saver()

    @lazy_property
    def loss(self):
        # print(tf.reshape(self.network, [self.conf.batch_size]).eval().shape)
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(self.network,
                                                                                [self.conf.batch_size]), self.target))
        # return -tf.reduce_sum(self.target * tf.log(self.network))

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.loss)

    @lazy_property
    def network(self):
        # net = self.conv2d(self.images, 64, [7, 7], activation=tf.nn.relu, name='nn_h0_conv')
        # net = self.conv2d(net, 128, [5, 5], activation=tf.nn.relu, name='nn_h2_conv')
        # net = self.conv2d(net, 256, [3, 3], activation=tf.nn.relu, name='nn_h3_conv')
        #
        # net = tf.reshape(net, [self.conf.batch_size, -1])
        # out = self.linear(net, 1, 'nn_h4_lin')

        df_dim = 64

        net = self.image_processing_layer(self.images)

        net = self.conv2d(net, df_dim, k_w=9, k_h=9, name='nn_h0_conv')
        net = tf.nn.relu(net)

        net = self.conv2d(net, df_dim * 2, k_w=7, k_h=7, name='nn_h1_conv')
        net = tf.nn.relu(net)

        net = self.conv2d(net, df_dim * 4, k_w=5, k_h=5, name='nn_h2_conv')
        net = tf.nn.relu(net)

        net = self.conv2d(net, df_dim * 8, k_w=3, k_h=3, name='nn_h3_conv')
        net = tf.nn.relu(net)

        net = tf.reshape(net, [self.conf.batch_size, -1])
        out = self.linear(net, 1, 'nn_h4_lin')

        return tf.nn.sigmoid(out)
