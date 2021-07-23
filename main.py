import os
from time import time

import numpy as np
import tensorflow as tf
from advstego.nn.image_utils import save_images

from advstego.nn.sgan import SGAN
from advstego.steganography.lsb_matching import LSBMatching
from advstego.utils import logger
from argparse import ArgumentParser

flags = tf.compat.v1.app.flags
flags.DEFINE_string('model_name', 'sgan', 'Name of trainable model')
flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_string('alpha', 0.5, 'G loss = alpha * fake_loss + (1 - alpha) * stego_loss')
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_2", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("checkpoint_dir_next", "checkpoint_after_10", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean('need_to_load', False, 'Need to load saved model')
flags.DEFINE_string('img_format', 'jpg', 'Format of input images')

flags.DEFINE_string('dataset_name', 'celebA', 'Dataset Name')
flags.DEFINE_string('summaries_dir', './tf_log_alpha_05', 'Directory fot TF to store logs')
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
        dcgan = SGAN(sess, LSBMatching, config=FLAGS,
                     image_size=FLAGS.image_size,
                     batch_size=FLAGS.batch_size)

        if FLAGS.is_train:
            dcgan.train(start_epoch=0)
        else:
            dcgan.load(FLAGS.checkpoint_dir, 50400) # 50400 50800

        # n_batches = 200
        #
        # z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
        #
        # for i in range(n_batches):
        #     np.random.seed(int(time()))
        #     z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
        #     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        #     save_images(samples, i, folder='/home/dvolkhonskiy/datasets/new/sgan_generated')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str,
                    dest='datapath', help='Path to CelebrityA dataset',
                    metavar='DATAPATH', default='./data/')
    options = parser.parse_args()
    flags.DEFINE_string('data', options.datapath+'/celebA', 'Dataset directory')
    tf.app.run()
