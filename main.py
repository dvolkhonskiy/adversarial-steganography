import os

import numpy as np
import tensorflow as tf
# from nn.conv_adv_net import ConvAdvNet
from steganography.algorithms.lsb_matching import LSBMatching
from nn.image_utils import save_images_to_one, save_images
from nn.stego_adv_net import StegoAdvNet
from utils.logger import logger
from time import strftime, gmtime

flags = tf.app.flags
flags.DEFINE_string('model_name', 'celeba', 'Name of trainable model')

flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_string('alpha', 0.7, 'G loss = alpha * fake_loss + (1 - alpha) * stego_loss')
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean('need_to_load', False, 'Need to load saved model')
flags.DEFINE_string('img_format', 'png', 'Format of input images')
# flags.DEFINE_string('data', '/home/dvolkhonskiy/datasets/lusn/bedroom_train_lmdb', 'Dataset directory')
flags.DEFINE_string('data', '/home/dvolkhonskiy/datasets/img_align_celeba', 'Dataset directory')
flags.DEFINE_string('dataset_name', 'celeba', 'Dataset Name')
# flags.DEFINE_string('data', '/home/dvolkhonskiy/datasets/lusn/data', 'Dataset directory')
# flags.DEFINE_string('data', '/home/dvolkhonskiy/datasets/bossbase', 'Dataset directory')
# flags.DEFINE_string('data', '/Users/dvolkhonskiy/Dropbox/science/deep-stego/nn/data/img_align_celeba', 'Dataset directory')
flags.DEFINE_string('summaries_dir', './tf_log', 'Directory fot TF to store logs')
FLAGS = flags.FLAGS


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
        dcgan = StegoAdvNet(sess, LSBMatching, config=FLAGS,
                            image_size=FLAGS.image_size,
                            batch_size=FLAGS.batch_size)

        if FLAGS.is_train:
            dcgan.train()
        else:
            dcgan.load(FLAGS.checkpoint_dir, 128000)

        n_batches = 200

        z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
        images = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

        for i in range(n_batches):
            z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, i, folder='/home/dvolkhonskiy/datasets/stego_celeb/gen_test')
        # save_images_to_one(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))

if __name__ == '__main__':
    tf.app.run()
