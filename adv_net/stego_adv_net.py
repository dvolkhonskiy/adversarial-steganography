import os
import numpy as np
import tensorflow as tf
from image_utils import *
import time
from layers import *
from glob import glob
from steganography.algorithms import BaseStego
from utils.logger import logger, log
import matplotlib.pyplot as plt
import seaborn as sns


class StegoAdvNet(object):
    def __init__(self, sess, stego_algorithm, image_size=108,
                 batch_size=64, sample_size=64, image_shape=(64, 64, 3),
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default'):
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

        self.stego_algorithm = stego_algorithm

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

        self.init_batch_norms()

        self.dataset_name = dataset_name

        self.images = tf.placeholder(tf.float32, [self.batch_size] + list(self.image_shape), name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + list(self.image_shape), name='sample_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.init_neural_networks()
        self.init_loss()
        self.init_vars()

        self.saver = tf.train.Saver()

    @log('Initializing neural networks')
    def init_neural_networks(self):
        # generator
        self.generator = self.generator_nn(self.z, train=True)

        # discriminator real/fake
        self.D_real = self.discriminator_real_fake_nn(self.images)

        self.D_stego = self.discriminator_stego_nn(self.stego_algorithm.encode(self.generator,
                                                                               np.random.randint(0, 2, 100),
                                                                               ))

        self.D_fake = self.discriminator_real_fake_nn(self.generator, reuse=True)

        # discriminator stego

        self.D_not_stego = self.discriminator_stego_nn(self.generator, reuse=True)
        # sampler
        self.sampler = self.generator_nn(self.z, train=False)

    @log('Initializing batch norms')
    def init_batch_norms(self):
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(self.batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(self.batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(self.batch_size, name='d_bn3')

        self.d_s_bn1 = batch_norm(self.batch_size, name='d_s_bn1')
        self.d_s_bn2 = batch_norm(self.batch_size, name='d_s_bn2')
        self.d_s_bn3 = batch_norm(self.batch_size, name='d_s_bn3')

        self.g_bn0 = batch_norm(self.batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(self.batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(self.batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(self.batch_size, name='g_bn3')

    @log('Initializing loss')
    def init_loss(self):
        # fake / real discriminator loss
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_real), self.D_real)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_fake), self.D_fake)
        self.d_fr_loss = self.d_loss_real + self.d_loss_fake

        # stego / non-stego discriminator

        self.d_loss_stego = binary_cross_entropy_with_logits(tf.ones_like(self.D_stego), self.D_stego)
        self.d_loss_nonstego = binary_cross_entropy_with_logits(tf.zeros_like(self.D_not_stego), self.D_not_stego)
        self.d_stego_loss_total = self.d_loss_stego + self.d_loss_nonstego

        # image generator loss
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_fake), self.D_fake)

    @log('Initializing variables')
    def init_vars(self):
        t_vars = tf.trainable_variables()

        self.d_r_f_vars = [var for var in t_vars if 'd_fr_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_s_n_vars = [var for var in t_vars if 'd_stego_' in var.name]

    @log('Training')
    def train(self, config):
        if config.need_to_load:
            self.load(config.checkpoint_dir)

        data = glob(os.path.join("./adv_net/data", config.dataset, "*.jpg"))
        # np.random.shuffle(data)

        d_r_f_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_fr_loss, var_list=self.d_r_f_vars)
        d_s_n_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_stego_loss_total, var_list=self.d_s_n_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate,  beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        batch_idxs = min(len(data), config.train_size) / config.batch_size

        for epoch in range(config.epoch):
            stego_losses, fake_real_losses, generator_losses = [], [], []

            for idx in range(0, int(batch_idxs)):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update discriminators network
                self.sess.run(d_r_f_optim, feed_dict={self.images: batch_images, self.z: batch_z})
                self.sess.run(d_s_n_optim, feed_dict={self.images: batch_images, self.z: batch_z})

                # Update generator network twice
                self.sess.run(g_optim, feed_dict={self.z: batch_z})
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})

                # todo encode here
                errD_stego = self.d_loss_stego.eval({self.z: batch_z})
                errD_n_stego = self.d_loss_nonstego.eval({self.z: batch_z})

                errG = self.g_loss.eval({self.z: batch_z})

                fake_real_losses.append(errD_fake + errD_stego)
                stego_losses.append(errD_stego + errD_n_stego)
                generator_losses.append(errG)

                logger.debug("[ITERATION] Epoch [%2d], iteration [%4d/%4d] time: %4.4f" %
                             (epoch, idx, batch_idxs, time.time() - start_time))
                logger.debug('[LOSS] Real/Fake: %.8f' % (errD_fake + errD_real))
                logger.debug('[LOSS] Stego/Non-Stego: %.8f' % (errD_stego + errD_n_stego))
                logger.debug('[LOSS] Generator: %.8f' % errG)

                counter += 1

                if np.mod(counter, 30) == 0:
                    samples, d_loss, g_loss, d_stego_loss = self.sess.run(
                        [self.sampler, self.d_fr_loss, self.g_loss,
                         self.d_stego_loss_total],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, 8], './samples/train_%s_%s.png' % (epoch, idx))

                    logger.info("[SAMPLE] d_fr_loss: %.8f, d_stego_loss: %.8f, g_loss: %.8f" % (d_loss, d_stego_loss, g_loss))

                    self.save(config.checkpoint_dir, counter)
                    self.plot_losses(epoch, stego_losses, fake_real_losses, generator_losses)

    @log('Plot losses for current epoch')
    def plot_losses(self, epoch, stego_losses, fake_real_losses, generator_losses, dir_to_save='losses'):
        plt.figure(figsize=(15, 10))

        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)
        every = 1

        plt.title("Training loss for %s epoch" % epoch)
        plt.xlabel("#iteration")
        plt.ylabel("Loss")
        plt.plot(stego_losses[::every], 'b')
        plt.plot(fake_real_losses[::every], 'r')
        plt.plot(generator_losses[::every], 'g')
        plt.legend(['Stego Loss', 'Fake/Real Loss', 'Generator Loss'])
        plt.savefig('%s/%s_epoch_loss.png' % (dir_to_save, epoch), dpi=50)

    def discriminator_real_fake_nn(self, img, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = leaky_relu(conv2d(img, self.df_dim, name='d_fr_h0_conv'))
        h1 = leaky_relu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_fr_h1_conv')))
        h2 = leaky_relu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_fr_h2_conv')))
        h3 = leaky_relu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_fr_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_fr_h3_lin')

        return tf.nn.sigmoid(h4)

    def discriminator_stego_nn(self, img, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = leaky_relu(conv2d(img, self.df_dim, name='d_stego_h0_conv'))
        h1 = leaky_relu(self.d_s_bn1(conv2d(h0, self.df_dim * 2, name='d_stego_h1_conv')))
        h2 = leaky_relu(self.d_s_bn2(conv2d(h1, self.df_dim * 4, name='d_stego_h2_conv')))
        h3 = leaky_relu(self.d_s_bn3(conv2d(h2, self.df_dim * 8, name='d_stego_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_stego_h3_lin')

        return tf.nn.sigmoid(h4)

    def generator_nn(self, noise, train=True):
        if not train:
            tf.get_variable_scope().reuse_variables()

        # project `noise` and reshape
        h0 = tf.reshape(linear(noise, self.gf_dim * 8 * 4 * 4, 'g_h0_lin'), [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=train))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=train))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=train))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=train))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    @log('Saving model')
    def save(self, checkpoint_dir, step):
        model_name = "Stego_DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    @log('Loading module')
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
