import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d as conv2d
from tensorflow.contrib.layers import fully_connected as linear
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits as cross_entropy

from .base_model import BaseModel
from .image_utils import get_image, save_images_to_one
from ..utils import logger, log

from tensorflow.contrib.layers import batch_norm as batch_norm


# from tensorflow.contrib.layers import conv2d_transpose as decon2d


class SGAN(BaseModel):
    def __init__(self, sess, stego_algorithm, config, image_size=64, batch_size=64, sample_size=64,
                 image_shape=(64, 64, 3), z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 dataset_name='celeb'):
        """
        :param sess: TensorFlow session
        :param stego_algorithm:
        :param image_size:
        :param batch_size: The size of batch. Should be specified before
        training.
        :param sample_size:
        :param image_shape: Shape of the images
        :param z_dim: Dimension of dim for Z. [100]
        :param gf_dim: Dimension of gen filters in first conv layer. [64]
        :param df_dim: Dimension of discrim filters in first conv layer. [64]
        :param gfc_dim: Dimension of gen untis for for fully connected layer. [1024]
        :param dfc_dim: Dimension of discrim units for fully connected layer. [1024]
        :param c_dim: Dimension of image color. [3]
        :param dataset_name: name of using dataset
        :return:
        """

        super().__init__(sess, config)

        self.stego_algorithm = stego_algorithm

        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.c_dim = c_dim

        self.images = tf.placeholder(tf.float32, [self.conf.batch_size] + list(self.image_shape), name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + list(self.image_shape),
                                            name='sample_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.init_neural_networks()
        self.init_loss()
        self.init_vars()

        self.init_saver()

    @log('Initializing neural networks')
    def init_neural_networks(self):
        # generator
        self.generator = self.generator_nn(self.z, train=True)

        # discriminator real/fake
        self.D_real  = self.discriminator(self.images)
        self.D_stego = self.eve(self.stego_algorithm().tf_encode(self.generator))

        # self.D_stego = self.discriminator_stego_nn(self.generator)
        self.D_fake = self.discriminator(self.generator, reuse=True)

        # discriminator stego
        self.D_not_stego = self.eve(self.generator, reuse=True)
        # sampler
        self.sampler = self.generator_nn(self.z, train=False)

    @log('Initializing loss')
    def init_loss(self):
        # fake / real discriminator loss
        self.d_loss_real = tf.reduce_mean(cross_entropy(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(cross_entropy(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.d_fr_loss = self.d_loss_real + self.d_loss_fake
        tf.summary.scalar('loss_d', self.d_fr_loss)

        # stego / non-stego discriminator

        self.d_loss_stego = tf.reduce_mean(cross_entropy(labels=tf.ones_like(self.D_stego), logits=self.D_stego))
        self.d_loss_nonstego = tf.reduce_mean(cross_entropy(labels=tf.zeros_like(self.D_not_stego), logits=self.D_not_stego))
        self.d_stego_loss_total = self.d_loss_stego + self.d_loss_nonstego
        tf.summary.scalar('loss_eve', self.d_stego_loss_total)

        self.g_loss_fake = tf.reduce_mean(cross_entropy(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.g_loss_stego = tf.reduce_mean(cross_entropy(labels=tf.ones_like(self.D_not_stego), logits=self.D_not_stego))

        self.g_loss = self.conf.alpha * self.g_loss_fake + (1. - self.conf.alpha) *self.g_loss_stego
        tf.summary.scalar('loss_g', self.g_loss)

    @log('Initializing variables')
    def init_vars(self):
        t_vars = tf.trainable_variables()

        self.d_fr_vars = [var for var in t_vars if 'd_fr_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_s_n_vars = [var for var in t_vars if 'd_s_' in var.name]

    @log('Training')
    def train(self):
        if self.conf.need_to_load:
            self.load(self.conf.checkpoint_dir)

        data = glob(os.path.join(self.conf.data, "*.%s" % self.conf.img_format))
        logger.info('Total amount of images: %s' % len(data))
        # np.random.shuffle(data)

        d_fr_optim = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        d_fr_optim = d_fr_optim.minimize(self.d_fr_loss, var_list=self.d_fr_vars)

        d_s_n_optim = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        d_s_n_optim = d_s_n_optim.minimize(self.d_stego_loss_total, var_list=self.d_s_n_vars)

        g_optim = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        g_optim = g_optim.minimize(self.g_loss, var_list=self.g_vars)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs_sgan', self.sess.graph)

        tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, need_transform=True) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 0
        start_time = time.time()
        batch_idxs = min(len(data), self.conf.train_size) / self.conf.batch_size

        logger.debug('Starting updating')
        for epoch in range(self.conf.epoch):
            stego_losses, fake_real_losses, generator_losses = [], [], []

            logger.info('Starting epoch %s' % epoch)

            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * self.conf.batch_size:(idx + 1) * self.conf.batch_size]
                batch = [get_image(batch_file, self.image_size, need_transform=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.conf.batch_size, self.z_dim]).astype(np.float32)

                out = self.sess.run([merged, d_fr_optim, d_s_n_optim, g_optim, g_optim], feed_dict={self.images: batch_images, self.z: batch_z})
                summary = out[0]

                train_writer.add_summary(summary, global_step=counter)

                # self.sess.run(d_s_n_optim, feed_dict={self.images: batch_images, self.z: batch_z})
                #
                # self.sess.run(g_optim, feed_dict={self.z: batch_z})
                # self.sess.run(g_optim, feed_dict={self.z: batch_z})

                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f" %
                             (epoch, idx, batch_idxs, time.time() - start_time))

                counter += 1

            self.save(self.conf.checkpoint_dir, counter)

            logger.info('Save samples')
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_fr_loss, self.g_loss_fake,
                 ],
                feed_dict={self.z: sample_z, self.images: sample_images}
            )
            save_images_to_one(samples, [8, 8], './samples/train_%s.png' % (epoch))

    def discriminator(self, img, reuse=False):
        with tf.variable_scope('D_network'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            net = batch_norm(img, center=True, scale=True, activation_fn=None, scope='d_fr_bn0')
            net = conv2d(net, self.df_dim, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h0_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_fr_bn1')
            net = conv2d(net, self.df_dim * 2, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h1_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_fr_bn2')
            net = conv2d(net, self.df_dim * 4, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h2_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_fr_bn3')
            net = conv2d(net, self.df_dim * 8, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h3_conv')

            net = tf.reshape(net, [self.conf.batch_size, -1])
            net = linear(net, 1, activation_fn=None, scope='d_fr_h4_lin',
                         weights_initializer=tf.random_normal_initializer(stddev=0.02))

            return net

    def eve(self, img, reuse=False):
        with tf.variable_scope('S_network'):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            net = img
            net = self.image_processing_layer(img)
            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_s_bn0')
            net = conv2d(net, self.df_dim, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h0_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_s_bn1')
            net = conv2d(net, self.df_dim * 2, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h1_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_s_bn2')
            net = conv2d(net, self.df_dim * 4, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h2_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_s_bn3')
            net = conv2d(net, self.df_dim * 8, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h3_conv')

            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='d_s_bn4')

            net = tf.reshape(net, [self.conf.batch_size, -1])
            net = linear(net, 1, activation_fn=tf.nn.sigmoid, scope='d_s_h4_lin',
                         weights_initializer=tf.random_normal_initializer(stddev=0.02))

            return net

    def generator_nn(self, noise, train=True):
        with tf.variable_scope('G_network'):
            if not train:
                tf.get_variable_scope().reuse_variables()

            net = linear(noise, self.gf_dim * 8 * 4 * 4, scope='g_h0_lin',
                         activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

            net = tf.reshape(net, [-1, 4, 4, self.gf_dim * 8])
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn0')
            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='g_bn0')
            net = tf.nn.relu(net)

            net = self.conv2d_transpose(net, [self.conf.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn1')
            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='g_bn1')
            net = tf.nn.relu(net)

            net = self.conv2d_transpose(net, [self.conf.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn2')
            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='g_bn2')
            net = tf.nn.relu(net)

            net = self.conv2d_transpose(net, [self.conf.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn3')
            net = batch_norm(net, center=True, scale=True, activation_fn=None, scope='g_bn3')
            net = tf.nn.relu(net)

            out = self.conv2d_transpose(net, [self.conf.batch_size, 64, 64, self.c_dim], name='g_out')

            return tf.nn.tanh(out)
