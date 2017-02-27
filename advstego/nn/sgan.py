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


# from tensorflow.contrib.layers import conv2d_transpose as decon2d


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


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

        self.d_bn0 = batch_norm(name='d_bn0')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.init_neural_networks()
        self.init_loss()
        self.init_vars()

        self.init_saver()

    @log('Initializing neural networks')
    def init_neural_networks(self):
        # generator
        self.generator = self.generator_nn(self.z, train=True)

        # discriminator real/fake
        self.D_real  = self.discriminator_real_fake_nn(self.images)
        self.D_stego = self.discriminator_stego_nn(self.stego_algorithm().tf_encode(self.generator))

        # self.D_stego = self.discriminator_stego_nn(self.generator)
        self.D_fake = self.discriminator_real_fake_nn(self.generator, reuse=True)

        # discriminator stego
        self.D_not_stego = self.discriminator_stego_nn(self.generator, reuse=True)
        # sampler
        self.sampler = self.generator_nn(self.z, train=False)

    @log('Initializing loss')
    def init_loss(self):
        # fake / real discriminator loss
        self.d_loss_real = tf.reduce_mean(cross_entropy(self.D_real, tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(cross_entropy(self.D_fake, tf.zeros_like(self.D_fake)))
        self.d_fr_loss = self.d_loss_real + self.d_loss_fake

        # stego / non-stego discriminator

        self.d_loss_stego = tf.reduce_mean(cross_entropy(tf.ones_like(self.D_stego), self.D_stego))
        self.d_loss_nonstego = tf.reduce_mean(cross_entropy(tf.zeros_like(self.D_not_stego), self.D_not_stego))
        self.d_stego_loss_total = self.d_loss_stego + self.d_loss_nonstego

        self.g_loss_fake = tf.reduce_mean(cross_entropy(self.D_fake, tf.ones_like(self.D_fake)))
        self.g_loss_stego = tf.reduce_mean(cross_entropy(tf.ones_like(self.D_not_stego), self.D_not_stego))

        self.g_loss = self.conf.alpha * self.g_loss_fake + (1. - self.conf.alpha) *self.g_loss_stego

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

        g_optim_fake = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        g_optim_fake = g_optim_fake.minimize(self.g_loss, var_list=self.g_vars)

        # g_optim_stego = tf.train.AdamOptimizer(0.000005, beta1=0.9)
        # g_optim_stego = g_optim_stego.minimize(self.g_loss_stego, var_list=self.g_vars)

        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter('./logs_sgan', self.sess.graph)

        tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, need_transform=True) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
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

                self.sess.run(d_fr_optim, feed_dict={self.images: batch_images, self.z: batch_z})
                self.sess.run(d_s_n_optim, feed_dict={self.images: batch_images, self.z: batch_z})

                self.sess.run(g_optim_fake, feed_dict={self.z: batch_z})
                self.sess.run(g_optim_fake, feed_dict={self.z: batch_z})

                # # if epoch > 5:
                # self.sess.run(g_optim_stego, feed_dict={self.z: batch_z})

                # errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                # errD_real = self.d_loss_real.eval({self.images: batch_images})
                #
                # errD_stego = self.d_loss_stego.eval({self.z: batch_z})
                # errD_n_stego = self.d_loss_nonstego.eval({self.z: batch_z})
                #
                # errG = self.g_loss.eval({self.z: batch_z})
                #
                # fake_real_losses.append(errD_fake + errD_stego)
                # stego_losses.append(errD_stego + errD_n_stego)
                # generator_losses.append(errG)
                #
                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f" %
                             (epoch, idx, batch_idxs, time.time() - start_time))
                # logger.debug('[LOSS] Real/Fake: %.8f' % (errD_fake + errD_real))
                # logger.debug('[LOSS] Stego/Non-Stego: %.8f' % (errD_stego + errD_n_stego))
                # logger.debug('[LOSS] Generator: %.8f' % errG)

                counter += 1

                if np.mod(counter, 1000) == 0:
                    self.save(self.conf.checkpoint_dir, counter)

                if np.mod(counter, 300) == 0:
                    logger.info('Save samples')
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_fr_loss, self.g_loss_fake,
                         ],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images_to_one(samples, [8, 8], './samples/train_%s_%s.png' % (epoch, idx))

    def discriminator_real_fake_nn(self, img, reuse=False):
        with tf.variable_scope('D_network'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # net = self.image_processing_layer(img)

            # net = self.batch_norm(net, scope='d_fr_bn0')
            net = self.d_bn0(img)
            net = conv2d(net, self.df_dim, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h0_conv')

            # net = self.batch_norm(net, scope='d_fr_bn1')
            net = self.d_bn1(net)
            net = conv2d(net, self.df_dim * 2, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h1_conv')

            # net = self.batch_norm(net, scope='d_fr_bn2')
            net = self.d_bn2(net)
            net = conv2d(net, self.df_dim * 4, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h2_conv')

            # net = self.batch_norm(net, scope='d_fr_bn3')
            net = self.d_bn3(net)
            net = conv2d(net, self.df_dim * 8, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_fr_h3_conv')

            # net = self.batch_norm(net, scope='d_fr_bn4')

            net = tf.reshape(net, [self.conf.batch_size, -1])
            net = linear(net, 1, activation_fn=None, scope='d_fr_h4_lin',
                         weights_initializer=tf.random_normal_initializer(stddev=0.02))

            return net

    def discriminator_stego_nn(self, img, reuse=False):
        with tf.variable_scope('S_network'):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            net = img
            net = self.image_processing_layer(img)
            net = self.batch_norm(net, scope='d_s_bn0')
            net = conv2d(net, self.df_dim, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h0_conv')

            net = self.batch_norm(net, scope='d_s_bn1')
            net = conv2d(net, self.df_dim * 2, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h1_conv')

            net = self.batch_norm(net, scope='d_s_bn2')
            net = conv2d(net, self.df_dim * 4, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h2_conv')

            net = self.batch_norm(net, scope='d_s_bn3')
            net = conv2d(net, self.df_dim * 8, kernel_size=[5, 5], stride=[2, 2],
                         activation_fn=self.leaky_relu, scope='d_s_h3_conv')

            net = self.batch_norm(net, scope='d_s_bn4')

            net = tf.reshape(net, [self.conf.batch_size, -1])
            net = linear(net, 1, activation_fn=tf.nn.sigmoid, scope='d_s_h4_lin',
                         weights_initializer=tf.random_normal_initializer(stddev=0.02))

            return net

    def generator_nn(self, noise, train=True):
        with tf.variable_scope('G_network'):
            if not train:
                tf.get_variable_scope().reuse_variables()

            gen = linear(noise, self.gf_dim * 8 * 4 * 4, scope='g_h0_lin',
                         activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

            gen = tf.reshape(gen, [-1, 4, 4, self.gf_dim * 8])
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn0')
            gen = self.g_bn0(gen, train=train)
            gen = tf.nn.relu(gen)

            gen = self.conv2d_transpose(gen, [self.conf.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn1')
            gen = self.g_bn1(gen, train=train)
            gen = tf.nn.relu(gen)

            gen = self.conv2d_transpose(gen, [self.conf.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn2')
            gen = self.g_bn2(gen, train=train)
            gen = tf.nn.relu(gen)

            gen = self.conv2d_transpose(gen, [self.conf.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
            # gen = self.batch_norm(gen, reuse=(not train), scope='g_bn3')
            gen = self.g_bn3(gen, train=train)
            gen = tf.nn.relu(gen)

            out = self.conv2d_transpose(gen, [self.conf.batch_size, 64, 64, self.c_dim], name='g_out')

            return tf.nn.tanh(out)
