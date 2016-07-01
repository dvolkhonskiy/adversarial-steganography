from PIL import Image
from glob import glob
import PIL
import shutil
import numpy as np


basewidth = 64
hsize = 64

files_to_move = glob('./img_align_celeba/*.png')

for _file in files_to_move:
    if np.random.random() > 0.1:
        print('%s to train' % _file)
        shutil.copy(_file, './stego_celeb/hugo_train')
    else:
        print('%s to test' % _file)
        shutil.copy(_file, './stego_celeb/hugo_test')
