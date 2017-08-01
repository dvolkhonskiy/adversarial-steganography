import functools
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d as conv2d
from tensorflow.contrib.layers import fully_connected as linear
from tensorflow.contrib.layers import batch_norm as batch_norm

from advstego.nn import BaseModel
from advstego.nn import get_image
from advstego.utils import logger, log


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class Steganalyzer(BaseModel):
    def __init__(self, config, sess, stego_algorithm, image_shape=(64, 64, 3), stego_name=None):
        super().__init__(sess, config)
        self.stego_algorithm = stego_algorithm
        self.image_shape = image_shape
        self.stego_name = stego_name

        self.images = tf.placeholder(tf.float32, [self.conf.batch_size] + list(self.image_shape))
        self.target = tf.placeholder(tf.float32, [self.conf.batch_size, 2])

        # if stego_name:
        self.data = self.get_images_names('%s/*.%s' % (config.data, self.conf.img_format))

        # init
        self.loss
        self.optimize
        self.network

    def get_targets(self, batch_files):
        get_tar = lambda x: int('stego_' in os.path.split(x)[-1])
        targets = np.array([get_tar(f) for f in batch_files], dtype=np.int32)
        out = np.zeros((self.conf.batch_size, 2), dtype=np.float32)
        out[range(targets.shape[0]), targets] = 1.
        # print(targets)
        return out

    @log('Training')
    def train(self, counter=1, gen_dirs=()):
        if self.conf.need_to_load:
            self.load(self.conf.checkpoint_dir, step=counter)

        data = self.data
        logger.info('Total amount of images: %s' % len(data))
        # np.random.shuffle(data)

        tf.initialize_all_variables().run()

        # counter = 1
        start_time = time.time()
        batch_idxs = min(len(data), self.conf.train_size) / self.conf.batch_size

        logger.debug('Starting updating')
        for epoch in range(self.conf.epoch):
            losses = []

            np.random.shuffle(data)

            logger.info('Starting epoch %s' % epoch)

            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]
                batch = [get_image(batch_file, self.conf.image_size)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_targets = self.get_targets(batch_files)

                _, loss = self.sess.run([self.optimize, self.loss], feed_dict={self.images: batch_images, self.target: batch_targets})

                losses.append(loss)

                # logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f, Loss: %8f, accuracy: %8f" %
                #              (epoch, idx, batch_idxs, time.time() - start_time, loss, stego_accuracy))

                counter += 1

            if epoch % 5 == 0 and epoch > 0:
                for gen_dir in gen_dirs:
                    acc, std = self.accuracy(n_files=-1, test_dir=gen_dir)
                    logger.info('[GEN_TEST] Folder %s, accuracy: %.4f, std: %.4f' % (gen_dir, acc, std))

            # SAVE after each epoch
            self.save(self.conf.checkpoint_dir, epoch)

    def get_accuracy(self, X_test, y_test):
        stego_answs = self.sess.run(self.network, feed_dict={self.images: X_test})
        stego_mistakes = tf.equal(tf.argmax(y_test, 1), tf.argmax(stego_answs, 1))

        return tf.reduce_mean(tf.cast(stego_mistakes, tf.float32)).eval()

    def accuracy(self, test_dir='test', abs=False, n_files=2 ** 12):
        # logger.info('[TEST], test data folder: %s, n_files: %s' % (test_dir, 2 * n_files))
        X_test = self.get_images_names('%s/*.%s' % (test_dir, self.conf.img_format), abs=abs)[:n_files]

        accuracies = []

        batch_idxs = min(len(X_test), self.conf.train_size) / self.conf.batch_size

        # logger.debug('Starting iteration')
        for idx in range(0, int(batch_idxs)):
            batch_files_stego = X_test[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]
            batch = [get_image(batch_file, self.conf.image_size) for batch_file in batch_files_stego]
            batch_images = np.array(batch).astype(np.float32)

            batch_targets = self.get_targets(batch_files_stego)

            accuracies.append(self.get_accuracy(batch_images, batch_targets))

        return np.mean(accuracies), np.std(accuracies)

    @lazy_property
    def saver(self):
        return tf.train.Saver()

    @lazy_property
    def loss(self):
        errs = tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.target)
        return tf.reduce_mean(errs)

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1).minimize(self.loss)

    @lazy_property
    def network(self):
        net = self.images

        net = self.image_processing_layer(net)

        def get_init():
            return tf.truncated_normal_initializer(stddev=0.02)

        net = conv2d(net, 10, [7, 7], activation_fn=tf.nn.relu, scope='conv1', weights_initializer=get_init())
        # net = batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, scope='d_bn1')
        net = conv2d(net, 20, [5, 5], activation_fn=tf.nn.relu, scope='conv2', weights_initializer=get_init())
        # net = batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, scope='d_bn2')
        net = tf.nn.max_pool(net, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')

        net = conv2d(net, 30, [3, 3], activation_fn=tf.nn.relu, scope='conv3', weights_initializer=get_init())
        # net = batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, scope='d_bn3')
        net = conv2d(net, 40, [3, 3], activation_fn=tf.nn.relu, scope='conv4', weights_initializer=get_init())
        # net = batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, scope='d_bn4')

        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

        net = tf.reshape(net, [self.conf.batch_size, -1])

        net = linear(net, 100, activation_fn=tf.nn.tanh, scope='FC1')
        out = linear(net, 2, activation_fn=tf.nn.softmax, scope='out')
        return out
