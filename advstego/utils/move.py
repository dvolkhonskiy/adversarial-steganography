from PIL import Image
from glob import glob
import PIL
import shutil
import numpy as np


basewidth = 64
hsize = 64

files_to_move = glob('/home/dvolkhonskiy/SGAN/code/data/10_seeds/*.png')

for _file in files_to_move[:50000]:
    if np.random.random() > 0.1:
        print('%s to train' % _file)
        shutil.move(_file, '/home/dvolkhonskiy/SGAN/code/data/10_seeds/train')
    else:
        print('%s to test' % _file)
        shutil.move(_file, '/home/dvolkhonskiy/SGAN/code/data/10_seeds/test')

