import os
from multiprocessing import Pool
from glob import glob
from PIL import Image
import sys
from subprocess import getoutput

sys.path.append('../')

from steganography import LSBMatching, LSB
from nn.image_utils import transform, imread
from scipy.misc import imsave


img_dir = '/home/dvolkhonskiy/datasets/generated_for_training/lsb_matching_test'
ext = '*.png'
algo = LSBMatching()


def delete_img(img):
    if 'stego_' in os.path.split(img)[-1] or 'empty_' in os.path.split(img)[-1]:
        print('removing %s' % img)
        os.remove(img)
        return True
    return False

        # try:
        #     scipy.misc.imread(img)
        # except OSError:
        #     os.remove(img)


def apply_stego(img):
    print('Embedding to %s' % img)
    path = os.path.split(img)
    info = algo.get_information(batch_size=1, len_of_text=450)[0]
    algo.encode(img, info, os.path.join(path[0], 'stego_' + path[-1]))


def resize_image(img):
    print('Resizing %s' % img)
    imsave(img, transform(imread(img), 108))


def apply_matlab_stego(img):
    return getoutput('octave ../matlab_stego/HUGO/run_one.m %s' % img)


p = Pool(8)

# delete
img_list = glob(os.path.join(img_dir, ext))
p.map(delete_img, img_list)


# resize
# img_list = glob(os.path.join(img_dir, ext))
# p.map(resize_image, img_list)


# embedding
img_list = glob(os.path.join(img_dir, ext))
p.map(apply_stego, img_list)

p.close()



