
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
from scipy.misc import imsave, imread

img_dir = '/home/dvolkhonskiy/datasets/stego_celeb/hugo_train'
ext = '*.png'


def main():
    img_list = glob(os.path.join(img_dir, ext))

    for i, img in enumerate(img_list):
        print('Processing %s' % img)

        # if 'stego_' in os.path.split(img)[-1] or 'empty_' in os.path.split(img)[-1]:
        #     print('removing')
        #     os.remove(img)
        try:
            scipy.misc.imread(img)
        except OSError:
            os.remove(img)


if __name__ == '__main__':
    main()
