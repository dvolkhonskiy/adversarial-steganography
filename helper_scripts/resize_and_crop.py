
"""
Crop and resize all images in folder to given size
"""
import os
from glob import glob
from PIL import Image


import sys
sys.path.append('../')

from nn.image_utils import transform, imread
import scipy.misc
from scipy.misc import imsave

img_dir = '/home/dvolkhonskiy/datasets/stego_celeb/test'
ext = '*.png'


def main():
    img_list = glob(os.path.join(img_dir, ext))

    for i, img in enumerate(img_list):
        print('Processing %s' % img)
        imsave(img, transform(imread(img), npx=64))

        if i % 100 == 0:
            print('Processed %s from %s' % (i, len(img_list)))


if __name__ == '__main__':
    main()
