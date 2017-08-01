import os
import sys
from glob import glob
from multiprocessing import Pool
from subprocess import getoutput
import numpy as np

sys.path.append('../../')

from advstego.steganography import LSBMatching
from advstego.nn import transform, imread
from scipy.misc import imsave


# img_dirs = [
#     '/home/dvolkhonskiy/SGAN/code/data/10_seeds/train',
#     '/home/dvolkhonskiy/SGAN/code/data/10_seeds/test',
#     '/home/dvolkhonskiy/SGAN/code/data/10_seeds/other_seeds',
#     '/home/dvolkhonskiy/SGAN/code/data/10_seeds/more_train',
#     '/home/dvolkhonskiy/SGAN/code/data/10_seeds/more_train_other_seeds',
# ]

# img_dirs = [
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/',
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_0',
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_1',
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_2',
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_3',
#             '/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_4',
#         ]

img_dirs = ['/home/dvolkhonskiy/SGAN/code/data/overtraining/plus_%s' % i for i in range(0, 66)]
img_dirs.append('/home/dvolkhonskiy/SGAN/code/data/overtraining/')


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
    # info = algo.get_information(batch_size=1, len_of_text=450)[0]
    info = np.random.randint(0, 2, 1638) # 0.4 bpp
    algo.encode(img, info, os.path.join(path[0], 'stego_' + path[-1]))


def resize_image(img):
    print('Resizing %s' % img)
    imsave(img, transform(imread(img), 108))


def apply_matlab_stego(img):
    # TODO doesn't work yet
    return getoutput('octave ../matlab_stego/HUGO/run_one.m %s' % img)


for dir in img_dirs:

    p = Pool(8)

    # delete
    # img_list = glob(os.path.join(dir, ext))
    # p.map(delete_img, img_list)


    # resize
    # img_list = glob(os.path.join(img_dir, ext))
    # p.map(resize_image, img_list)


    # embedding
    img_list = glob(os.path.join(dir, ext))
    p.map(apply_stego, img_list)

    p.close()



