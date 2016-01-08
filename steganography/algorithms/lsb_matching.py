from steganography.algorithms import BaseStego
from PIL import Image
import numpy as np


class LSBMatching(BaseStego):
    def __init__(self):
        super().__init__()

    @staticmethod
    def encode(container, information, stego='stego.png'):
        img = Image.open(container)
        width, height = img.size
        img_matr = np.asarray(img)
        img_matr.setflags(write=True)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        information = information + ''.join(map(str, BaseStego.DELIMITER))
        for i, bit in enumerate(information):
            bit = int(bit)
            if bit == 0 and red_ch[i] & 1 == 1:
                red_ch[i] -= 1
            elif bit == 1 and red_ch[i] & 1 == 0:
                red_ch[i] += 1

        img_matr[:, :, 2] = red_ch.reshape((height, width))

        Image.fromarray(img_matr).save(stego)

    @staticmethod
    def decode(container):
        img = Image.open(container)
        img_matr = np.asarray(img)
        img_matr.setflags(write=True)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        delim_len = len(BaseStego.DELIMITER)

        information = np.array([], dtype=int)
        for pixel in red_ch:
            information = np.append(information, [pixel & 1])

            if information.shape[0] > delim_len and np.array_equiv(information[-delim_len:], BaseStego.DELIMITER):
                break

        information = information[:-delim_len]

        return ''.join(map(str, information))
