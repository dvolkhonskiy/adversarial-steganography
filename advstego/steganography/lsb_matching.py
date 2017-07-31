import numpy as np
import tensorflow as tf
from PIL import Image

# from advstego.steganography import BaseStego
from advstego.utils import logger

class BaseStego:
    DELIMITER = np.ones(100, dtype=int)  # TODO hidden info ends with 1, then decoder skip it

    def __init__(self):
        pass

    @staticmethod
    def tf_encode(container):
        raise NotImplementedError

    @staticmethod
    def encode(container, information):
        raise NotImplementedError

    @staticmethod
    def decode(container):
        raise NotImplementedError



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
        with tf.variable_scope('Stego'):

            n, width, height, chan = tuple(map(int, container._shape))

            info = np.random.randint(0, 2, (n, 1638))

            mask = np.zeros(list(container.get_shape()))

            print('Num of images: %s' % n)
            for img_idx in range(n):
                print(img_idx)

                for i, bit in enumerate(info[img_idx]):
                    ind, jnd = i // width, i - width * (i // width)

                    if tf.to_int32(container[img_idx, ind, jnd, 0]) * 127.5 % 2 != bit:
                        if np.random.randint(0, 2) == 0:
                            # tf.assign_sub(container[img_idx, ind, jnd, 0], 1)
                            mask[img_idx, ind, jnd, 0] += 1/256.
                        else:
                            # tf.assign_add(container[img_idx, ind, jnd, 0], 1)
                            mask[img_idx, ind, jnd, 0] -= 1/256.

            logger.debug('Finish encoding')
            return tf.add(container, mask)

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

        red_ch = img_matr[:, :, 0].reshape((1, -1))[0]

        information = np.append(information, BaseStego.DELIMITER)
        for i, bit in enumerate(information):

            if bit != red_ch[i] & 1:
                if np.random.randint(0, 2) == 0:
                    red_ch[i] -= 1
                else:
                    red_ch[i] += 1

        img_matr[:, :, 0] = red_ch.reshape((height, width))

        Image.fromarray(img_matr).save(stego)

    @staticmethod
    def decode(container):
        img = Image.open(container)
        img_matr = np.asarray(img)

        red_ch = img_matr[:, :, 0].reshape((1, -1))[0]

        delim_len = len(BaseStego.DELIMITER)

        info = np.array([], dtype=int)
        for pixel in red_ch:
            info = np.append(info, [pixel & 1])

            if info.shape[0] > delim_len and np.array_equiv(info[-delim_len:], BaseStego.DELIMITER):
                break

        info = info[:-delim_len]

        return ''.join(map(str, info))
