import os
from time import time

import numpy as np
import tensorflow as tf
from advstego.nn.image_utils import save_images

from advstego.nn.sgan import SGAN
from advstego.steganography.lsb_matching import LSBMatching
from advstego.utils import logger

flags = tf.app.flags
flags.DEFINE_string('model_name', 'sgan', 'Name of trainable model')
flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_string('alpha', 0.9, 'G loss = alpha * fake_loss + (1 - alpha) * stego_loss')
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_alpha_05", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean('need_to_load', False, 'Need to load saved model')
flags.DEFINE_string('img_format', 'png', 'Format of input images')
flags.DEFINE_string('data', './data', 'Dataset directory')
flags.DEFINE_string('dataset_name', 'celeba', 'Dataset Name')
flags.DEFINE_string('summaries_dir', './tf_log', 'Directory fot TF to store logs')
FLAGS = flags.FLAGS

set_of_seeds = list(range(0, 1001, 100))

set_of_seeds_test = list(range(2000, 3001, 100))


def main(_):
    logger.info('====================================================')
    logger.info('===================NEW EXPERIMENT===================')
    logger.info('====================================================')

    logger.info(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = SGAN(sess, LSBMatching, config=FLAGS,
                     image_size=FLAGS.image_size,
                     batch_size=FLAGS.batch_size)

        if FLAGS.is_train:
            dcgan.train()
        else:
            dcgan.load(FLAGS.checkpoint_dir, 31650) # 31650 28485

        n_batches = 199

        z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))

        print('Seed for sampling', set_of_seeds)

        for i in range(n_batches):
            if i % 100 == 0:
                print('Iteration', i)
            current_seed = set_of_seeds[int(i / 20)]
            np.random.seed(current_seed)
            z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, i, folder='./data/10_seeds/more_train')

        for i in range(n_batches):
            if i % 100 == 0:
                print('Iteration', i)
            current_seed = set_of_seeds_test[int(i / 20)]
            np.random.seed(current_seed)
            z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, i, folder='./data/10_seeds/more_train_other_seeds')

if __name__ == '__main__':
    tf.app.run()
