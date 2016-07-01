import numpy as np
from steganography.text import texts
from utils.string_utils import str_to_bin

class BaseStego:
    DELIMITER = np.ones(100, dtype=int)  # TODO hidden info ends with 1, then decoder skip it

    def __init__(self):
        pass

    @staticmethod
    def get_information(batch_size=64, len_of_text=500):
        start_idxes = np.random.randint(0, len(texts) - 1, batch_size)
        return [str_to_bin(texts[start_idx:start_idx+len_of_text]) for start_idx in start_idxes]

    @staticmethod
    def tf_encode(container):
        raise NotImplementedError

    @staticmethod
    def encode(container, information):
        raise NotImplementedError

    @staticmethod
    def decode(container):
        raise NotImplementedError
