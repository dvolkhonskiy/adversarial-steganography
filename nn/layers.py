import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# from nn.image_utils import *
from utils.logger import logger


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, batch_size, epsilon=1e-5, momentum=0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                         initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
                                        initializer=tf.constant_initializer(0.))

            mean, variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)


def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) + (1. - logits) * tf.log(1. - targets + eps)))


