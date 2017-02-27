import numpy as np


def str_to_bin(text):
    bins = ''.join(format(x, 'b') for x in bytearray(text, 'utf8'))
    return np.array(list(bins), dtype=np.int32)


def bin_to_str(bin_arr):
    pass