import os
import numpy as np

import tensorflow as tf
from image_utils import *
import time

from layers import *
from glob import glob
from utils.logger import logger


class ConvAdvNet(object):
    def __init__(self, sess, image_size=108,
                 batch_size=64, sample_size=64, image_shape=(64, 64, 3),
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name

        self.images = tf.placeholder(tf.float32, [self.batch_size] + list(self.image_shape),
                                     name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + list(self.image_shape),
                                            name='sample_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim],  name='z')

        self.G = self.generator(self.z)
        self.D = self.discriminator(self.images)

        self.sampler = self.sampler(self.z)
        self.D_ = self.discriminator(self.G, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.need_to_load:
            self.load(config.checkpoint_dir)

        data = glob(os.path.join("./adv_net/data", config.dataset, "*.jpg"))
        # np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        batch_idxs = min(len(data), config.train_size) / config.batch_size

        for epoch in range(config.epoch):
            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D_real network
                self.sess.run(d_optim, feed_dict={self.images: batch_images, self.z: batch_z})

                # Update generator network
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                # Run g_optim twice to make sure that d_fr_loss does not go to zero (different from paper)
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})
                logger.debug("errD:", errD_fake + errD_real, "errD_fake:", errD_fake, "errD_real", errD_real, "errG", errG)

                counter += 1
                logger.debug("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fr_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 30) == 0:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_%s_%s.png' % (epoch, idx))
                    logger.info("[Sample] d_fr_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 30) == 0:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, input, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = leaky_relu(conv2d(input, self.df_dim, name='d_h0_conv'))
        h1 = leaky_relu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
        h2 = leaky_relu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
        h3 = leaky_relu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4)

    def generator(self, noise):
        # project `noise` and reshape
        h0 = tf.reshape(linear(noise, self.gf_dim * 8 * 4 * 4, 'g_h0_lin'), [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    def sampler(self, z):
        tf.get_variable_scope().reuse_variables()

        # project `z` and reshape
        h0 = tf.reshape(linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin'), [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        logger.info(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
