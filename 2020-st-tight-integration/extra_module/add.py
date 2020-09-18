import tensorflow as tf
from TFNetworkLayer import _ConcatInputLayer, CopyLayer
import numpy as np


class AddOneHotToTime(CopyLayer):
  layer_class = "addonehot"

  def __init__(self, position=0, repeat=1, vocab_size=30000, **kwargs):
    """
    :param float|str prefix: either some constant or another layer
    :param int repeat: how often to repeat the prefix
    """
    super(AddOneHotToTime, self).__init__(**kwargs)
    assert self.output.time_dim_axis is not None
    assert isinstance(position, (int)), "Idice needs to be Integer"
    c = tf.one_hot(position, depth=vocab_size, dtype=self.output.dtype)
    shape = [((self.output.batch_shape[i] or tf.shape(self.output.placeholder)[i])
              if (i != self.output.time_dim_axis)
              else repeat)
             for i in range(self.output.batch_ndim)]
    x = tf.ones(shape, dtype=self.output.dtype)
    self.output.placeholder = tf.concat([self.output.placeholder, x * c], axis=self.output.time_dim_axis)
    self.output.size_placeholder[self.output.time_dim_axis_excluding_batch] += repeat

class AddZero(CopyLayer):
  layer_class = "addzero"

  def __init__(self, repeat=1, **kwargs):
    """
    :param float|str prefix: either some constant or another layer
    :param int repeat: how often to repeat the prefix
    """
    super(AddZero, self).__init__(**kwargs)
    assert self.output.time_dim_axis is not None
    c = tf.zeros([1.0], dtype=self.output.dtype)
    shape = [((self.output.batch_shape[i] or tf.shape(self.output.placeholder)[i])
              if (i != self.output.time_dim_axis)
              else repeat)
             for i in range(self.output.batch_ndim)]
    x = tf.ones(shape, dtype=self.output.dtype)
    self.output.placeholder = tf.concat([self.output.placeholder, x * c], axis=self.output.time_dim_axis)
    self.output.size_placeholder[self.output.time_dim_axis_excluding_batch] += repeat

