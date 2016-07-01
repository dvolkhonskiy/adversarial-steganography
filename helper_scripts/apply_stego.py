"""
Apply stego algorithm to all images in given folder
"""

import os
from glob import glob
from PIL import Image

import sys
sys.path.append('../')

from steganography import LSBMatching, LSB

img_dir = '/home/dvolkhonskiy/datasets/stego_celeb/lsb_train'
ext = '*.png'
algo = LSB()


def main():
    img_list = glob(os.path.join(img_dir, ext))
    info = algo.get_information(batch_size=len(img_list), len_of_text=450)

    for i, img in enumerate(img_list):
        path = os.path.split(img)
        print('Processing %s' % img)
        algo.encode(img, info[i], os.path.join(path[0], 'stego_' + path[-1]))
        # Image.open(img).save(os.path.join(path[0], 'empty_' + path[-1]))

        # os.remove(img)

        if i % 100 == 0:
            print('Processed %s from %s' % (i, len(img_list)))


if __name__ == '__main__':
    main()
