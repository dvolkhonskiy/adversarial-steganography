from steganography.algorithms import BaseStego
from PIL import Image
import numpy as np
import tensorflow as tf

from utils.logger import logger, log


class LSBMatching(BaseStego):
    def __init__(self):
        super().__init__()

    @staticmethod
    def tf_encode(container):
        """
        LSB matching algorithm (+-1 embedding)
        :param container: tf tensor shape (batch_size, width, height, chan)
        :param information: array with int bits
        :param stego: name of image with hidden message
        """

        n, width, height, chan = tuple(map(int, container._shape))

        information = BaseStego.get_information(n, 50)
        logger.debug('Information to hide', information)

        print('Num of images: %s' % n)
        for img_idx in range(n):
            print(img_idx)

            for i, bit in enumerate(information[img_idx]):
                ind, jnd = i // width, i - width * (i // width)

                if tf.to_int32(container[img_idx, ind, jnd, 0]) % 2 != bit:
                    if np.random.randint(0, 2) == 0:
                        tf.sub(container[img_idx, ind, jnd, 0], 1)
                    else:
                        tf.add(container[img_idx, ind, jnd, 0], 1)

        logger.debug('Finish encoding')
        return container

    @staticmethod
    def encode(container, information, stego='stego.png'):
        """
        LSB matching algorithm (+-1 embedding)
        :param container: path to image container
        :param information: array with int bits
        :param stego: name of image with hidden message
        """
        pass

    @staticmethod
    def decode(container):
        pass
