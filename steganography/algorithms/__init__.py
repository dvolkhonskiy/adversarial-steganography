import numpy as np


class BaseStego:
    DELIMITER = np.ones(100, dtype=int)  # TODO hidden info ends with 1, then decoder skip it

    def __init__(self):
        pass

    @staticmethod
    def encode(container, information):
        raise NotImplementedError

    @staticmethod
    def decode(container):
        raise NotImplementedError
