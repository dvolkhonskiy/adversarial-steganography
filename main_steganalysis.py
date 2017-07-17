import os

import numpy as np
import tensorflow as tf

from advstego.nn.steganalyzer import Steganalyzer
from advstego.steganography.lsb_matching import LSBMatching
from advstego.utils import logger

flags = tf.app.flags
flags.DEFINE_string('model_name', 'stego_lsb_matching', 'Name of trainable model')
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.000005, "Learning rate of for adam [0.000005]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean('need_to_load', False, 'Need to load saved model')
flags.DEFINE_string('img_format', 'png', 'Format of input images')
flags.DEFINE_string('data', '/home/dvolkhonskiy/SGAN/code/data/10_seeds', 'Dataset directory')
flags.DEFINE_string('dataset_name', 'celeba', 'Dataset Name')
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
        steganalisys = Steganalyzer(config=FLAGS, sess=sess, stego_algorithm=LSBMatching)

        if FLAGS.is_train:
            steganalisys.train(counter=1, gen_dirs=[
                '/home/dvolkhonskiy/SGAN/code/data/10_seeds/test',
                '/home/dvolkhonskiy/SGAN/code/data/10_seeds/other_seeds',
                '/home/dvolkhonskiy/SGAN/code/data/10_seeds/more_train',
                '/home/dvolkhonskiy/SGAN/code/data/10_seeds/more_train_other_seeds',
            ])
        else:
            # steganalisys.load(FLAGS.checkpoint_dir, step=28481) # LSB matching clf
            steganalisys.load(FLAGS.checkpoint_dir, step=122114)

        # print('ACCURACY:::::::::%s' % steganalisys.accuracy(test_dir='gen_test_more_train', n_files=-1))

if __name__ == '__main__':
    tf.app.run()
