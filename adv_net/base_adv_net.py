import os
import numpy as np

import tensorflow as tf
from image_utils import *
import time

from layers import *
from glob import glob


# TODO BASE class for adversarial convolution neural network

class BaseAdvNet(object):
    def __init__(self):
        raise NotImplementedError

    def train(self, config):
        raise NotImplementedError

    def save(self, checkpoint_dir, step):
        raise NotImplementedError

    def load(self, checkpoint_dir):
        raise NotImplementedError
