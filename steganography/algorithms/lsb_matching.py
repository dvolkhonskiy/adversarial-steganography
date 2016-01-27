from steganography.algorithms import BaseStego
from PIL import Image
import numpy as np
from numba import jit


class LSBMatching(BaseStego):
    def __init__(self):
        super().__init__()

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

        print(information)
        print(BaseStego.DELIMITER)
        information = np.append([information, BaseStego.DELIMITER])
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
