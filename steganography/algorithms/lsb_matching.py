from steganography.algorithms import BaseStego
from PIL import Image
import numpy as np
import tensorflow as tf

from utils.logger import logger, log


class LSBMatching(BaseStego):
    def __init__(self):
        super().__init__()

    @log('Encoding LSB matching')
    def tf_encode(self, container):
        """
        LSB matching algorithm (+-1 embedding)
        :param container: tf tensor shape (batch_size, width, height, chan)
        :param information: array with int bits
        :param stego: name of image with hidden message
        """

        n, width, height, chan = tuple(map(int, container._shape))

        information = self.get_information(n, 50)
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
        img = Image.open(container)
        width, height = img.size
        img_matr = np.asarray(img)
        img_matr.setflags(write=True)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        information = np.append(information, BaseStego.DELIMITER)
        for i, bit in enumerate(information):

            if bit == 0 and red_ch[i] & 1 == 1 or bit == 1 and red_ch[i] & 1 == 0:
                if np.random.randint(0, 2) == 0:
                    red_ch[i] -= 1
                else:
                    red_ch[i] += 1

        img_matr[:, :, 2] = red_ch.reshape((height, width))

        Image.fromarray(img_matr).save(stego)

    @staticmethod
    def decode(container):
        img = Image.open(container)
        img_matr = np.asarray(img)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        delim_len = len(BaseStego.DELIMITER)

        info = np.array([], dtype=int)
        for pixel in red_ch:
            info = np.append(info, [pixel & 1])

            if info.shape[0] > delim_len and np.array_equiv(info[-delim_len:], BaseStego.DELIMITER):
                break

        info = info[:-delim_len]

        return ''.join(map(str, info))
