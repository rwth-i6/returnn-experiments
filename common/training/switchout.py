from ..asr.specaugment import random_mask


def switchout_target(self, source, targetb_blank_idx: int,
                     target_num_labels: int, time_factor: int = 6,
                     switchout_prob: float = 0.05,
                     switchout_blank_prob: float = 0.5, **kwargs):
  """Switchout. It takes as input a batch of outputs and returns a switchout version of it.
  Usage:
    {
    "class": "eval", "from": "output", "eval": switchout_target,
    "eval_local": {"_targetb_blank_idx": target.blankid, "_target_num_labels": target.get_numclasses()},
    "initial_output": 0
    }
  Args:
      source ([Data]): (B,T,)
      targetb_blank_idx (int): index for blank label
      target_num_labels (int): size of vocab
      time_factor (int, optional): Defaults to 6.
      switchout_prob (float, optional): Probability for switchout
      switchout_blank_prob (float, optional): Probab to choose blank if switchout is happening.

  Returns:
      (tf.tensor): switched out input (B,T,)
  """
  from returnn.tf.util.basic import where_bc
  from returnn.tf.compat import v1 as tf
  network = self.network
  data = source(0, as_data=True)
  assert data.is_batch_major  # just not implemented otherwise
  x = data.placeholder

  def get_switched():
    x_ = x
    shape = tf.shape(x)
    take_rnd_mask = tf.less(tf.random_uniform(shape=shape, minval=0., maxval=1.), switchout_prob)
    take_blank_mask = tf.less(tf.random_uniform(shape=shape, minval=0., maxval=1.), switchout_blank_prob)
    rnd_label = tf.random_uniform(shape=shape, minval=0, maxval=target_num_labels, dtype=tf.int32)
    rnd_label = where_bc(take_blank_mask, targetb_blank_idx, rnd_label)
    x_ = where_bc(take_rnd_mask, rnd_label, x_)
    x_ = random_mask(
      x_, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
      min_num=0, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // (50 // time_factor), 1),
      max_dims=20 // time_factor,
      mask_value=targetb_blank_idx)
    # x_ = tf.Print(x_, ["switch", x[0], "to", x_[0]], summarize=100)
    return x_
  x = network.cond_on_train(get_switched, lambda: x)
  return x
