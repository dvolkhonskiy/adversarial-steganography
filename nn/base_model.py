import os
import numpy as np
import tensorflow as tf
from nn.image_utils import get_image, save_images_to_one
import time
from nn.layers import *
from glob import glob
from utils.logger import logger, log
import tensorflow.contrib.skflow as skflow


class BaseModel:
    def __init__(self, sess, config):
        self.sess = sess
        self.conf = config
        self.tf_init_rnd_norm = tf.random_normal_initializer

    def init_saver(self):
        self.saver = tf.train.Saver()

    def get_images_names(self, _format):
        return glob(os.path.join(self.conf.data, _format))

    @staticmethod
    def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', shape=[k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
            return conv

    # @staticmethod
    # def conv2d(X, n_filters, filter_shape, bias=False, activation=None, name='conv2d'):
    #     with tf.variable_scope(name):
    #         return skflow.ops.conv2d(X, n_filters=n_filters,
    #                                  filter_shape=filter_shape, bias=bias,
    #                                  activation=activation)

    @staticmethod
    def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    @staticmethod
    def leaky_relu(x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    @staticmethod
    def linear(input_, output_size, scope=None, stddev=0.02):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=stddev))
            return tf.matmul(input_, matrix)

    @log('Initializing neural networks')
    def init_neural_networks(self):
        raise NotImplementedError

    @log('Initializing loss')
    def init_loss(self):
        raise NotImplementedError

    @log('Train')
    def train(self):
        raise NotImplementedError

    def save(self, checkpoint_dir, step):

        model_dir = "%s_%s" % (self.conf.model_name, self.conf.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ckpt_name = '%s_%s.ckpt' % (self.conf.model_name, step)

        logger.info('[SAVING] step: %s, name: %s' % (step, ckpt_name))
        self.saver.save(self.sess, os.path.join(checkpoint_dir, ckpt_name), global_step=step)

    @log('Loading module')
    def load(self, checkpoint_dir, step):
        model_dir = "%s_%s" % (self.conf.model_name, self.conf.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)


        try:
            ckpt_name = '%s_%s.ckpt-%s' % (self.conf.model_name, step, step)

            logger.info('[LOADING] step: %s, name: %s' % (step, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        except Exception as e:
            logger.debug(e)
            ckpt_name = 'StegoDCGAN-%s' % (step)

            logger.info('[LOADING] step: %s, name: %s' % (step, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
