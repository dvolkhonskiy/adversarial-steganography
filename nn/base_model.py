import os
import numpy as np
import tensorflow as tf
from nn.image_utils import get_image, save_images_to_one
import time
from glob import glob
from utils.logger import logger, log


class BaseModel:
    def __init__(self, sess, config):
        self.sess = sess
        self.conf = config
        self.tf_init_rnd_norm = tf.random_normal_initializer

    def init_saver(self):
        self.saver = tf.train.Saver()

    def get_images_names(self, _format, abs=False):
        if abs:
            return glob(_format)
        return glob(os.path.join(self.conf.data, _format))

    @staticmethod
    def batch_norm(x, scope='batch_norm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            shape = x.get_shape().as_list()

            mean, var = tf.nn.moments(x, axes=[0])

            gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
            beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))

            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

    @staticmethod
    def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    @staticmethod
    def leaky_relu(x, alpha=0.2, name="lrelu"):
        with tf.variable_scope(name):
            return tf.maximum(alpha * x, x)

    def image_processing_layer(self, X):
        K = 1 / 12. * tf.constant([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=tf.float32)

        kernel = tf.pack([K, K, K])
        kernel = tf.pack([kernel, kernel, kernel])

        return tf.nn.conv2d(X, tf.transpose(kernel, [2, 3, 0, 1]), [1, 1, 1, 1], padding='SAME')

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
