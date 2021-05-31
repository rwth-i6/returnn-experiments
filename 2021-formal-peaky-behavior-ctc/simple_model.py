#!/usr/bin/env python3

"""
This is the script to run synthetic examples.
These are most of the simulations in the paper.

It provides functions for:

- generate input
- generate output (as target + CTC topo, or directly via FSA, see :mod:`fst_utils`)
- losses (original CTC loss and variations for any FSA, optional with prior, etc)
- models (bias, memory, FFNN, any custom (e.g. BLSTM), model-free, generative)
- parameter initialization (zero -> uniform by default)
- training options (optimizer, num steps, learning rate, etc)
- training loop with logging and statistics

See :func:`model` and :func:`run` and :func:`main`.

It provides several entry points, such as all functions with `run_` prefix,
which you can call like `./simple_model.py run_ffnn_sum`.
These correspond to all the simulations in the paper, and more.
Without arguments, it will directly call :func:`run` with default options.

It should run with both TF1 and TF2 (using non-eager/graph mode).
It needs RETURNN as dependency.

Probably you want to play around with the default parameters.

"""

import sys
import os
import argparse
import numpy
from pprint import pprint
import tensorflow as tf
from fst_utils import Fsa, get_std_fsa_1label
from returnn.tf.native_op import ctc_loss, ctc_loss_viterbi, fast_viterbi, get_ctc_fsa_fast_bw, NativeLstm2
from returnn.tf.util.basic import expand_dims_unbroadcast, add_scaled_noise_to_gradients, safe_log
from returnn.tf.util.basic import custom_gradient, where_bc, dot


tf1 = tf.compat.v1
tf1.disable_eager_execution()
tf1.disable_v2_tensorshape()
tf1.disable_control_flow_v2()
tf = tf1


def apply_weight_noise(p, scale):
  """
  :param tf.Variable|tf.Tensor p:
  :param float scale:
  :rtype: tf.Tensor
  """
  if not scale:
    return p
  return p + tf.truncated_normal(p.shape, stddev=scale)


def get_summaries_dict_from_bytes(s):
  """
  :param bytes s: via session.run(tf.summary.merge_all())
  :rtype: dict[str,float]
  """
  sum_vs = tf.summary.Summary()
  sum_vs.ParseFromString(s)
  d = {}
  for v in sum_vs.value:
    field_names = [k.name for k, _ in v.ListFields()]
    if "simple_value" in field_names:
      d[v.tag] = v.simple_value
      continue
    if "tensor" in field_names:
      assert v.tensor.tensor_content
      x = tf.make_ndarray(v.tensor)
      assert isinstance(x, numpy.ndarray)
      if x.ndim == 1:
        x = x.tolist()
      d[v.tag] = x
      continue
  return d


def get_scaled_grads_by_1_m_prior(x):
  """
  :param tf.Tensor x: e.g. logits, shape (...,dim)
  :return: x, but different grad
  """
  dim = x.get_shape().dims[-1].value
  assert dim is not None, "x shape %s [-1] must be defined" % x.get_shape()

  def forward_op(x_):
    return x_

  def grad_op(op_, out_grad):
    # out_grad == softmax(logits) - bw
    bw = tf.nn.softmax(x) - out_grad

    avg_bw = tf.reduce_mean(tf.squeeze(bw, axis=1), axis=0)  # (dim,)
    avg_bw.set_shape((dim,))
    tf.summary.tensor_summary("avg_bw", avg_bw)

    return out_grad * (1. - avg_bw)

  op = custom_gradient.register([tf.float32], op=forward_op, grad_op=grad_op)
  y = op(x)
  y.set_shape(x.get_shape())
  return y


def full_sum_loss(logits, logits_seq_lens, logits_time_major, targets, targets_seq_lens):
  """
  Similar to :func:`tf.nn.ctc_loss`.
  We use our :func:`fast_baum_welch`.
  Also see :class:`FastBaumWelchLoss`.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
  :param tf.Tensor logits_seq_lens: shape (batch,) of int32|int64
  :param bool logits_time_major:
  :param tf.Tensor|Fsa targets: batch-major, [batch,time]
  :param tf.Tensor|None targets_seq_lens: (batch,)
  :return: loss, shape (batch,)
  :rtype: tf.Tensor
  """
  if isinstance(targets, Fsa):
    assert logits_time_major
    # Warning: logits_seq_lens ignored currently...
    return -targets.tf_get_full_sum(logits=tf.nn.log_softmax(logits))
  return ctc_loss(
    logits=logits, logits_seq_lens=logits_seq_lens, logits_time_major=logits_time_major,
    targets=targets, targets_seq_lens=targets_seq_lens)


def full_sum_loss_no_renorm(logits, logits_seq_lens, logits_time_major, targets, targets_seq_lens):
  """
  Similar to :func:`tf.nn.ctc_loss`.
  We use our :func:`fast_baum_welch`.
  Also see :class:`FastBaumWelchLoss`.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
  :param tf.Tensor logits_seq_lens: shape (batch,) of int32|int64
  :param bool logits_time_major:
  :param tf.Tensor|Fsa targets: batch-major, [batch,time]
  :param tf.Tensor|None targets_seq_lens: (batch,)
  :return: loss, shape (batch,)
  :rtype: tf.Tensor
  """
  assert isinstance(targets, Fsa)  # not implemented otherwise
  assert logits_time_major
  # Warning: logits_seq_lens ignored currently...
  return -targets.tf_get_full_sum(logits=logits)


def full_sum_loss_with_prior(logits, logits_seq_lens, logits_time_major, targets, targets_seq_lens):
  """
  Similar to :func:`tf.nn.ctc_loss`.
  We use our :func:`fast_baum_welch`.
  Also see :class:`FastBaumWelchLoss`.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
  :param tf.Tensor logits_seq_lens: shape (batch,) of int32|int64
  :param bool logits_time_major:
  :param tf.Tensor|Fsa targets: batch-major, [batch,time]
  :param tf.Tensor|None targets_seq_lens: (batch,)
  :return: loss, shape (batch,)
  :rtype: tf.Tensor
  """
  assert isinstance(targets, Fsa)  # not implemented otherwise
  assert logits_time_major
  # Warning: logits_seq_lens ignored currently...

  log_sm = tf.nn.log_softmax(logits)
  sm = tf.exp(log_sm)
  avg_sm = tf.reduce_mean(tf.squeeze(sm, axis=1), axis=0)
  scores = log_sm - safe_log(avg_sm)
  return -targets.tf_get_full_sum(logits=scores)


def full_sum_loss_with_stop_grad_prior(logits, logits_seq_lens, logits_time_major, targets, targets_seq_lens):
  """
  Similar to :func:`tf.nn.ctc_loss`.
  We use our :func:`fast_baum_welch`.
  Also see :class:`FastBaumWelchLoss`.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
  :param tf.Tensor logits_seq_lens: shape (batch,) of int32|int64
  :param bool logits_time_major:
  :param tf.Tensor targets: batch-major, [batch,time]
  :param tf.Tensor targets_seq_lens: (batch,)
  :return: loss, shape (batch,)
  :rtype: tf.Tensor
  """
  assert logits.get_shape().ndims == 3 and logits.get_shape().dims[-1].value
  dim = logits.get_shape().dims[-1].value
  if not logits_time_major:
    logits = tf.transpose(logits, [1, 0, 2])  # (time,batch,dim)

  # No need for stop_gradient here; we will control it via custom_gradient.
  log_sm = tf.nn.log_softmax(logits)  # (time,batch,dim)
  sm = tf.exp(log_sm)
  # Note: Not the correct masking here. Should be fixed, but does not matter for the demo here.
  avg_sm = tf.reduce_mean(sm, axis=0, keep_dims=True)  # (1,1,dim)
  am_scores = log_sm - safe_log(avg_sm)

  from returnn.TFUtil import sequence_mask_time_major
  seq_mask = sequence_mask_time_major(logits_seq_lens)  # (time,batch)

  from returnn.TFNativeOp import get_ctc_fsa_fast_bw, fast_baum_welch
  edges, weights, start_end_states = get_ctc_fsa_fast_bw(
    targets=targets, seq_lens=targets_seq_lens, blank_idx=dim - 1)
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=-am_scores,  # -log space
    float_idx=seq_mask,
    edges=edges, weights=weights, start_end_states=start_end_states)
  loss = obs_scores[0]  # (batch,)
  n_batch = tf.shape(loss)[0]
  bw = tf.exp(-fwdbwd)  # (time,batch,dim). fwdbwd in -log space
  grad_x = where_bc(tf.expand_dims(seq_mask, 2), sm - bw, 0.0)  # (time,batch,dim)
  loss = tf.reshape(loss, [1, n_batch, 1])  # (1,batch,1), such that we can broadcast to logits/grad_x
  loss = custom_gradient.generic_loss_and_error_signal(loss=loss, x=logits, grad_x=grad_x)
  loss = tf.reshape(loss, [n_batch])
  return loss


def generate_input(input_seq, num_frames, num_classes):
  """
  :param list[int] input_seq:
  :param int num_frames:
  :param int num_classes: excluding blank
  :rtype: tf.Tensor
  :return: (num_frames,dim), dim = num_classes + 1
  """
  assert input_seq
  assert len(input_seq) == num_frames
  assert all([0 <= x < num_classes + 1 for x in input_seq])
  dim = num_classes + 1  # including blank
  input_seq_t = tf.constant(input_seq)
  return tf.one_hot(input_seq_t, dim, 1., 0.)


def model(
      num_classes, target_seq,
      model_type,
      num_frames=None,
      input_seq=None,
      scale_sm_by_prior=False,
      loss_type="sum",
      init_type="zero", rnd_scale=1., rnd_seed=42, blank_bias_init=None,
      opt_class=tf1.train.GradientDescentOptimizer, learning_rate=0.1,
      logits_time_dropout=0, grad_noise=0, weight_noise=0,
      scale_update_inv_param_size=False,
      update_exact=False,
      scale_grads_by_1_m_prior=False):
  """
  :param int num_classes: except blank
  :param int|None num_frames:
  :param list[int]|Fsa|None target_seq:
  :param str model_type: "bias" or "mem" or "mem+bias", "ff", "blstm"
  :param list[int]|None input_seq: if given, length should be like num_frames. will use tf.one_hot
  :param bool scale_sm_by_prior:
  :param str loss_type: "sum" or "max"
  :param str init_type: "zero" or "rnd_normal"
  :param float rnd_scale:
  :param int rnd_seed:
  :param float|None blank_bias_init:
  :param type opt_class:
  :param float learning_rate:
  :param bool scale_update_inv_param_size:
  :param float logits_time_dropout:
  :param float grad_noise:
  :param float weight_noise:
  :param bool update_exact:
  :param bool scale_grads_by_1_m_prior:
  :return:
  """
  if num_frames is None:
    assert input_seq is not None
    num_frames = len(input_seq)
  dim = num_classes + 1
  rnd = numpy.random.RandomState(rnd_seed)
  tf1.set_random_seed(rnd.randint(0, 2 ** 16))
  if init_type == "zero":
    init_func = numpy.zeros
  elif init_type == "rnd_normal":
    def init_func(shape, dtype):
      return rnd.normal(size=shape, scale=rnd_scale).astype(dtype)
  elif init_type == "rnd_uniform":    
    def init_func(shape, dtype):
      return rnd.uniform(size=shape, low=-rnd_scale, high=rnd_scale).astype(dtype)
  elif init_type == "identity":
    def init_func(shape, dtype):
      if len(shape) == 2 and shape[0] == shape[1]:
        return numpy.eye(shape[0], dtype=dtype)
      return numpy.zeros(shape=shape, dtype=dtype)
  else:
    raise ValueError("invalid init_type %r" % (init_type,))
  if loss_type == "sum":
    loss_func = full_sum_loss
  elif loss_type == "gen_sum":
    loss_func = full_sum_loss_no_renorm
  elif loss_type == "sum_with_prior":
    loss_func = full_sum_loss_with_prior
  elif loss_type == "sum_with_prior_sg":
    # Very similar to `scale_sm_by_prior` option, but no renorm afterwards.
    loss_func = full_sum_loss_with_stop_grad_prior
  elif loss_type == "max":
    loss_func = ctc_loss_viterbi
  elif loss_type == "sum+max":
    def loss_func(**kwargs):
      return (ctc_loss(**kwargs) + ctc_loss_viterbi(**kwargs)) * 0.5
  else:
    raise ValueError("invalid loss_type %r" % (loss_type,))
  global_step = tf1.train.get_or_create_global_step()
  mem = None

  if model_type == "bias":
    bias_init = init_func((dim,), dtype="float32")
    if blank_bias_init is not None:
      bias_init[-1] = blank_bias_init
    bias = tf.get_variable("bias", shape=(dim,), initializer=tf.constant_initializer(value=bias_init))
    params = [bias]
    bias = apply_weight_noise(bias, weight_noise)
    logits = tf.expand_dims(expand_dims_unbroadcast(bias, axis=0, dim=num_frames), axis=1)  # (time,batch,dim)

  elif model_type == "mem":
    mem_init = init_func((num_frames, dim), dtype="float32")
    if blank_bias_init is not None:
      mem_init[:, -1] = blank_bias_init
    mem = tf.get_variable("mem", shape=(num_frames, dim), initializer=tf.constant_initializer(value=mem_init))
    params = [mem]
    mem = apply_weight_noise(mem, weight_noise)
    logits = tf.expand_dims(mem, axis=1)  # (time,batch,dim)

  elif model_type == "mem+bias":
    mem_init = init_func((num_frames, dim), dtype="float32")
    if blank_bias_init is not None:
      mem_init[:, -1] = blank_bias_init
    mem = tf.get_variable("mem", shape=(num_frames, dim), initializer=tf.constant_initializer(value=mem_init))
    bias_init = numpy.zeros((dim,), dtype="float32")
    if blank_bias_init is not None:
      bias_init[-1] = blank_bias_init
    bias = tf.get_variable("bias", shape=(dim,), initializer=tf.constant_initializer(value=bias_init))
    params = [mem, bias]
    mem = apply_weight_noise(mem, weight_noise)
    bias = apply_weight_noise(bias, weight_noise)
    logits_bias = tf.expand_dims(expand_dims_unbroadcast(bias, axis=0, dim=num_frames), axis=1)  # (time,batch,dim)
    logits = tf.expand_dims(mem, axis=1) + logits_bias  # (time,batch,dim)

  elif model_type == "model_free":
    assert dim == 2  # currently not implemented otherwise
    input_symbols = sorted(set(input_seq))
    param_init = init_func((len(input_symbols),), dtype="float32")
    param = tf.get_variable(  # single variable for all input symbols; plot_loss_grad_map can nicely plot this
      "param", shape=(len(input_symbols),), initializer=tf.constant_initializer(value=param_init))
    params = [param]
    input_seq_tensors = []
    for x in input_seq:
      p_idx = input_symbols.index(x)
      tensor = [param[x], -param[x]]
      if dim == len(input_symbols) == 2 and p_idx == 1:
        # Just for somewhat nicer / more consistent plotting to swap this around.
        # Really somewhat arbitrary, but does not matter anyway.
        tensor = [-param[x], param[x]]
      input_seq_tensors.append(tensor)
    logits = tf.convert_to_tensor(input_seq_tensors)  # (time,dim)
    logits.set_shape((len(input_seq), dim))
    logits = tf.expand_dims(logits, axis=1)  # (time,batch,dim)

  elif model_type == "gen_model_free":
    input_symbols = sorted(set(input_seq))
    assert len(input_symbols) == 2  # currently not implemented otherwise
    param_init = init_func((dim,), dtype="float32")
    param = tf.get_variable(  # single variable for all input symbols; plot_loss_grad_map can nicely plot this
      "param", shape=(dim,), initializer=tf.constant_initializer(value=param_init))
    params = [param]
    log_probs = []
    for i in range(dim):
      i_logits = [param[i], -param[i]]
      if i == 1 and dim == 2:
        # Just for somewhat nicer / more consistent plotting to swap this around.
        # Really somewhat arbitrary, but does not matter anyway.
        i_logits = [-param[i], param[i]]
      log_probs.append(tf.nn.log_softmax(i_logits))
    input_seq_tensors = []
    for x in input_seq:
      p_idx = input_symbols.index(x)
      tensor = [log_probs[i][p_idx] for i in range(dim)]
      input_seq_tensors.append(tensor)
    logits = tf.convert_to_tensor(input_seq_tensors)  # (time,dim)
    logits.set_shape((len(input_seq), dim))
    logits = tf.expand_dims(logits, axis=1)  # (time,batch,dim)

  elif isinstance(model_type, (dict, list)):  # generic NN
    mem = None
    if isinstance(model_type, list):
      model_type = {"layers": model_type}
    layers = model_type.get("layers", [])
    assert isinstance(layers, list)
    params = []
    x = generate_input(input_seq=input_seq, num_frames=num_frames, num_classes=num_classes)
    n_batch = 1
    x = tf.expand_dims(x, axis=1)  # (T,B,D)
    index = tf.ones([num_frames, n_batch])

    for i, layer in enumerate(layers):
      if isinstance(layer, str):
        layer = {"class": layer}
      assert isinstance(layer, dict)
      dim = layer.get("dim", max(num_classes * 2, 10))
      layer_class = layer["class"]

      if layer_class == "linear":
        shape = (x.shape[-1].value, dim)
        mat_init = init_func(shape, dtype="float32")
        mat = tf.get_variable("W%i" % i, shape=shape, initializer=tf.constant_initializer(value=mat_init))
        mat = apply_weight_noise(mat, weight_noise)
        x = dot(x, mat)
        if layer.get("bias", model_type.get("bias", True)):
          bias_init = init_func((dim,), dtype="float32")
          bias = tf.get_variable("b%i" % i, shape=(dim,), initializer=tf.constant_initializer(value=bias_init))
          bias = apply_weight_noise(bias, weight_noise)
          x = x + bias
        x.set_shape((num_frames, 1, dim))
        act = layer.get("act", "relu")
        if act:
          x = getattr(tf.nn, act)(x)

      elif layer_class == "blstm":
        shape = (x.shape[-1].value, dim * 4)
        xs = []
        for d in (-1, 1):
          with tf.variable_scope("blstm%i_%s" % (i, {-1: "bwd", 1: "fwd"}[d])):
            x_ = x
            mat_init = init_func(shape, dtype="float32")
            mat = tf.get_variable("W_ff", shape=shape, initializer=tf.constant_initializer(value=mat_init))
            mat = apply_weight_noise(mat, weight_noise)
            x_ = dot(x_, mat)
            if layer.get("bias", model_type.get("bias", True)):
              bias_init = init_func(shape[-1:], dtype="float32")
              bias = tf.get_variable("b", shape=shape[-1:], initializer=tf.constant_initializer(value=bias_init))
              bias = apply_weight_noise(bias, weight_noise)
              x_ = x_ + bias
            cell = NativeLstm2(n_hidden=dim, n_input_dim=shape[0], step=d)
            x_, _ = cell(
              x_, index,
              recurrent_weights_initializer=tf.constant_initializer(
                value=init_func((dim, dim * 4), dtype="float32")))
            xs.append(x_)
        x = tf.concat(axis=2, values=xs)  # [T,B,D*2]

      else:
        raise ValueError("invalid layer %i %r in model %r" % (i, layer, model_type))

    shape = (x.shape[-1].value, num_classes + 1)
    mat_init = init_func(shape, dtype="float32")
    mat = tf1.get_variable("W_final", shape=shape, initializer=tf.constant_initializer(value=mat_init))
    mat = apply_weight_noise(mat, weight_noise)
    x = dot(x, mat)
    if model_type.get("bias", True):
      bias_init = init_func(shape[-1:], dtype="float32")
      bias = tf1.get_variable("b_final", shape=shape[-1:], initializer=tf.constant_initializer(value=bias_init))
      bias = apply_weight_noise(bias, weight_noise)
      x = x + bias
    logits = x

    for p in tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES):
      if p not in params:
        params.append(p)

  else:
    raise ValueError("invalid model_type %r" % (model_type,))

  logits.set_shape((num_frames, 1, num_classes + 1))
  if logits_time_dropout:
    logits = tf.nn.dropout(
      logits, noise_shape=[num_frames, 1, 1],
      rate=tf.where(tf.equal(global_step % 2, 0), logits_time_dropout, 0.))
  if scale_sm_by_prior:
    # such that we can rescale by prior, norm them now
    logits -= tf.stop_gradient(tf.reduce_logsumexp(logits, axis=-1, keep_dims=True))
    sm = tf.exp(logits)
    avg_sm = tf.reduce_mean(sm, axis=0, keep_dims=True)  # (1,1,dim)
    logits -= tf.stop_gradient(safe_log(avg_sm))
  if scale_grads_by_1_m_prior:
    logits = get_scaled_grads_by_1_m_prior(logits)
  logits_seq_len = tf.convert_to_tensor([num_frames])  # (batch,)
  if target_seq is None:
    target_seq = list(range(num_classes))
  if isinstance(model_type, str) and model_type.startswith("gen_"):  # e.g. "gen_model_free"
    am_scores = logits
  else:
    am_scores = tf.nn.log_softmax(logits)
  if isinstance(target_seq, Fsa):
    pass
  else:
    assert len(target_seq) <= num_frames  # and that even might not be enough, e.g. for repeating entries
  targets_seq_len = None
  if isinstance(target_seq, Fsa):
    targets = target_seq
    viterbi, _ = targets.tf_get_best_alignment(logits=am_scores)
  else:
    targets = tf.convert_to_tensor([target_seq])  # (batch,time)
    targets_seq_len = tf.convert_to_tensor([len(target_seq)])  # (batch,)
    edges, weights, start_end_states = get_ctc_fsa_fast_bw(
      targets=targets, seq_lens=targets_seq_len, blank_idx=num_classes)
    viterbi, _ = fast_viterbi(
      am_scores=am_scores, am_seq_len=logits_seq_len,
      edges=edges, weights=weights, start_end_states=start_end_states)
    if input_seq:
      fer = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(viterbi[:,0], input_seq), tf.int32)), tf.float32) / tf.cast(num_frames, tf.float32)
      tf.summary.scalar("fer_viterbi_to_ref", fer)
      fer = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits[:,0],axis=-1,output_type=tf.int32), input_seq), tf.int32)), tf.float32) / tf.cast(num_frames, tf.float32)
      tf.summary.scalar("fer_softmax_to_ref", fer)
    fer = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits[:,0],axis=-1,output_type=tf.int32), viterbi[:,0]), tf.int32)), tf.float32) / tf.cast(num_frames, tf.float32)
    tf.summary.scalar("fer_softmax_to_viterbi", fer)
  assert isinstance(targets, (tf.Tensor, Fsa))
  loss = loss_func(
    logits=logits, logits_seq_lens=logits_seq_len, logits_time_major=True,
    targets=targets, targets_seq_lens=targets_seq_len)
  loss = tf.reduce_mean(loss)
  tf.summary.scalar("loss", loss)
  grads = tf.gradients(ys=[loss], xs=[logits] + params)
  logits_grad, param_grads = grads[0], grads[1:]
  # logits_grad == softmax(logits) - baum_welch
  baum_welch = tf.nn.softmax(logits) - logits_grad
  loss_bw = full_sum_loss(
    logits=safe_log(baum_welch), logits_seq_lens=logits_seq_len, logits_time_major=True,
    targets=targets, targets_seq_lens=targets_seq_len)
  tf.summary.scalar("loss_bw", tf.reduce_mean(loss_bw))
  assert len(params) == len(param_grads)
  for param, grad in zip(params, param_grads):
    assert grad is not None, "no grad for param %r?" % param
  grads_l2 = tf.reduce_sum([tf.nn.l2_loss(v) for v in param_grads])
  tf.summary.scalar("grad_norm", grads_l2)
  opt = opt_class(learning_rate=learning_rate)
  assert isinstance(opt, tf1.train.Optimizer)
  grads_and_vars = [(g, v) for (g, v) in zip(param_grads, params)]
  if grad_noise:
    grads_and_vars = add_scaled_noise_to_gradients(grads_and_vars, grad_noise)
  if scale_update_inv_param_size:
    max_num_elements = max([param.get_shape().num_elements() for param in params])
    grads_and_vars = [(g * (float(v.get_shape().num_elements()) / max_num_elements), v) for (g, v) in grads_and_vars]
  if update_exact:
    assert model_type == "mem"
    update_op = tf1.assign(mem, safe_log(tf.squeeze(baum_welch, axis=1)))
  else:
    update_op = opt.apply_gradients(grads_and_vars)
  with tf.control_dependencies([update_op]):
    update_op = tf1.assign_add(global_step, 1)
  return loss, logits, baum_welch, viterbi, update_op, params, param_grads


def run(num_steps=100, file=sys.stdout, log_params=True, fast=False, init_params=None, **kwargs):
  with tf.Graph().as_default():
    loss, logits, baum_welch, viterbi, update_ops, params, param_grads = model(**kwargs)
    summaries = tf1.summary.merge_all()
    sm = tf.nn.softmax(logits)
    assert isinstance(sm, tf.Tensor)
    nframes = sm.get_shape().dims[0].value
    dim = sm.get_shape().dims[-1].value
    assert dim is not None
    blank_label = dim - 1
    avg_sm = tf.reduce_mean(tf.squeeze(sm, axis=1), axis=0)  # (dim,)
    avg_bw = tf.reduce_mean(tf.squeeze(baum_welch, axis=1), axis=0)  # (dim,)
    min_p_blank = tf.reduce_min(sm[:, :, blank_label])
    argmax = tf.argmax(sm, axis=-1)  # (time,batch)
    num_blanks = tf.reduce_sum(tf1.to_int32(tf.equal(argmax, blank_label)))
    if viterbi is not None:
      num_blanks_viterbi = tf.reduce_sum(tf1.to_int32(tf.equal(viterbi, blank_label)))
    else:
      viterbi = tf.constant([[-1]])  # just some dummy
      num_blanks_viterbi = tf.constant(-1)
    loss_values = []
    with tf1.Session() as session:
      session.run(tf1.global_variables_initializer())
      if init_params:
        assert len(init_params) == len(params)
        for param, value in zip(params, init_params):
          param.load(value, session=session)
      for i in range(num_steps):
        detailed_log = i in [0, num_steps - 1]
        if (detailed_log and not fast) or log_params:
          pv = session.run(params)
        if log_params:
          for param, value in zip(params, pv):
            assert isinstance(param, tf.Variable)
            print("%s:" % param.name[:-2], file=file)
            with numpy.printoptions(threshold=None if detailed_log else 10):
              print(value, file=file)
        if detailed_log or not fast:
          lv, grad_values, smv, asmv, mpb, nbv, nbvv, bwv, abwv, vitv, sum_vs, _ = session.run(
            (loss, param_grads, sm, avg_sm, min_p_blank, num_blanks, num_blanks_viterbi, baum_welch, avg_bw, viterbi, summaries, update_ops))
        else:
          lv, _ = session.run((loss, update_ops))
        loss_values.append(lv)
        print("step %i: loss %f" % (i, lv), file=file)
        if detailed_log or not fast:
          sum_vs = get_summaries_dict_from_bytes(sum_vs)
          print("summaries:", sum_vs, file=file)
          print("avg softmax:", file=file)
          with numpy.printoptions(suppress=True, precision=2):
            print(asmv, file=file)
          print("avg Baum-Welch:", file=file)
          with numpy.printoptions(suppress=True, precision=2):
            print(abwv, file=file)
          print("Min_t p(blank|x):", mpb)
          print("num blanks argmax softmax:", nbv, "/", nframes, file=file)
          print("num blanks Viterbi:", nbvv, "/", nframes, file=file)
        if detailed_log:
          print("softmax:", file=file)
          assert smv.shape[1] == 1
          pprint(smv[:, 0].tolist(), width=120, stream=file)
          if not fast:
            print("Baum-Welch:", file=file)
            assert bwv.shape[1] == 1
            print(bwv[:, 0], file=file)
          print("Viterbi:", file=file)
          assert vitv.shape[1] == 1
          print(list(vitv[:, 0]), file=file)
        if log_params:
          for param, value in zip(params, grad_values):
            assert isinstance(param, tf.Variable)
            print("%s grad:" % param.name[:-2], file=file)
            with numpy.printoptions(threshold=10):
              print(value, file=file)
        elif detailed_log:
          print("params:", params, file=file)
      if fast:
        pv = session.run(params)
    return locals()


def get_loss_grad_map(value_range=(0., 1.), steps=11, **kwargs):
  xs = numpy.linspace(value_range[0], value_range[1], steps)
  ys = xs
  xi, yi = numpy.meshgrid(xs, ys)
  lossi = numpy.zeros_like(xi)
  grad_xi = numpy.zeros_like(xi)
  grad_yi = numpy.zeros_like(xi)
  assert xi.shape == yi.shape == grad_xi.shape == grad_yi.shape
  with tf.Graph().as_default():
    loss, logits, baum_welch, viterbi, update_ops, params, param_grads = model(**kwargs)
    with tf.Session() as session:
      assert len(params) == 1
      param, = params
      assert isinstance(param, tf.Variable)
      print("var:", param.name)
      assert param.shape.as_list() == [2]  # we want to get a 2D map
      for idx in numpy.ndindex(xi.shape):
        param_v = numpy.array([xi[idx], yi[idx]])
        param.load(param_v, session=session)
        loss_v, param_grads_v = session.run((loss, param_grads))
        lossi[idx] = loss_v
        for param, value in zip(params, param_grads_v):
          # print("value", param_v, "grad", value)
          grad_xi[idx], grad_yi[idx] = -value  # negative grad, because grad descent

  return xi, yi, lossi, grad_xi, grad_yi


def get_loss_grad_single(param_value, **kwargs):
  """
  :param numpy.ndarray|list[float] param_value:
  """
  param_value = numpy.asarray(param_value)
  with tf.Graph().as_default():
    loss, logits, baum_welch, viterbi, update_ops, params, param_grads = model(**kwargs)
    with tf.Session() as session:
      assert len(params) == 1
      param, = params
      assert isinstance(param, tf.Variable)
      print("var:", param.name)
      assert param.shape.as_list() == list(param_value.shape)
      param.load(param_value, session=session)
      loss_v, param_grads_v, baum_welch_v = session.run((loss, param_grads, tf.squeeze(baum_welch, axis=1)))
      assert len(param_grads_v) == len(params) == 1
      return loss_v, -param_grads_v[0], baum_welch_v  # negative grad, because grad descent


_N = 4  # n = 3 is not peaky, n = 4 is peaky
DefaultOpts = dict(
  loss_type="sum",
  model_type="mem",
  #model_type={"bias": False},
  #model_type=["blstm", "blstm", "blstm"],
  num_frames=None,
  #input_seq=[3] * 20 + [0] * 15 + [1] * 10 + [3] * 10 + [2] * 20 + [3] * 25,
  #input_seq=[3] * 10 + [0] * 25 + [1] * 25 + [3] * 5 + [2] * 25 + [3] * 10,
  #num_classes=3, target_seq=[0, 1, 2],
  num_classes=1, target_seq=[0], input_seq=[1] * _N + [0] * 2 * _N + [1] * _N,
  #loss_type="sum_with_prior",
  #model_type="mem+bias",
  #num_classes=1, target_seq=[0],
  init_type="zero",
  #init_type="identity",
  #init_type="rnd_normal",
  #rnd_scale=1.,
  rnd_scale=0.01,
  rnd_seed=1, blank_bias_init=None,
  opt_class=tf1.train.AdamOptimizer, learning_rate=0.1,
  #learning_rate=0.0001,
  #learning_rate=1.,
  grad_noise=0., weight_noise=0., logits_time_dropout=0.,
  scale_update_inv_param_size=False,
  # scale_sm_by_prior=True,  # or: loss_type="sum_with_prior"
  # scale_grads_by_1_m_prior=True,
  # update_exact=True  # in practice, cannot do that, so more interesting to not use it
)


def avg_blank_prob_hist(**kwargs):
  """
  Multiple runs with different random seeds.
  (Only makes sense if the initialization is random.)
  Collects histogram of the blank softmax output.
  """
  opts = DefaultOpts.copy()
  opts.update(kwargs)
  opts.update(dict(num_steps=50, learning_rate=0.1, file=open(os.devnull, "w")))
  values = []
  for i in range(200):
    opts.update(dict(rnd_seed=i))
    vs = run(**opts)
    b = vs["asmv"][-1]
    losses = vs["loss_values"]
    gnv = vs["sum_vs"]["grad_norm"]
    loss_bw = vs["sum_vs"]["loss_bw"]
    fer = vs["sum_vs"].get("fer_viterbi_to_ref", None)
    nbvv, nframes = vs["nbvv"], vs["nframes"]
    print(
      i, ": blank", b,
      "loss:", losses[0], "->", losses[-1],
      "final fer:", fer,
      "grad norm:", gnv,
      "loss_bw:", loss_bw,
      "num blanks (Viterbi):", nbvv, "/", nframes)
    values.append(b)
    hist, _ = numpy.histogram(values, bins=numpy.linspace(0., 1., num=21))
    print("hist:", hist)


def run_peaky_ctc_vs_sil():
  import fst_utils
  fsa_blank = fst_utils.get_std_fsa_3label_blank()
  fsa_sil = fst_utils.get_std_fsa_3label_sil()
  #fsa_blank = fst_utils.get_std_fsa_4label_2words_blank()
  #fsa_sil = fst_utils.get_std_fsa_4label_2words_sil()
  num_labels = len(fsa_sil.get_labels())
  n_ = 10
  input_seq = [4] * 2 * n_ + [1] * n_ + [2] * 3 * n_ + [3] * 2 * n_ + [4] * 2 * n_  # see plot_align
  # input_seq = [5] * n_ + [1] * 3 * n_ + [2] * 3 * n_ + [3] * 3 * n_ + [5] * n_ + [4] * 3 * n_ + [5] * n_
  input_seq = [x - 1 for x in input_seq]
  # count_by_label = {i: len([x for x in input_seq if x == i]) for i in range(num_labels)}
  # assert len(set(count_by_label.values())) == 1  # all same
  opts = dict(
    loss_type="sum",
    #model_type="mem",
    model_type={},  # 1-layer FFNN + bias
    #model_type={"bias": False},  # 1-layer FFNN - bias. note: this does not converge to peaky, depending on the example
    #model_type={"layers": [{"class": "blstm", "dim": 10, "bias": True}], "bias": True},
    log_params=False, fast=True,
    num_frames=None,
    num_classes=num_labels - 1,  # except sil/blank
    input_seq=input_seq,
    #init_type="rnd_uniform",  # important for BLSTM
    init_type="zero",
    #init_type="identity",
    rnd_scale=0.01,
    rnd_seed=1,
    opt_class=tf.train.AdamOptimizer,
    learning_rate=0.1,
    #learning_rate=1.,
    num_steps=50,
  )
  for fsa in [
    fsa_blank,
    fsa_sil
  ]:
    opts["target_seq"] = fsa
    run(**opts)


def _run_ratio_t_n(n_):
  # n_ is factor in T. T = 10 * n_ by construction below
  target_seq = [0, 1, 2]  # N = 3
  input_seq = [4] * 2 * n_ + [1] * n_ + [2] * 3 * n_ + [3] * 2 * n_ + [4] * 2 * n_  # see plot_align
  # input_seq = [5] * n_ + [1] * 3 * n_ + [2] * 3 * n_ + [3] * 3 * n_ + [5] * n_ + [4] * 3 * n_ + [5] * n_
  input_seq = [x - 1 for x in input_seq]
  cv_target_seq = [2, 1]
  cv_input_seq = [3] * n_ + [2] * 2 * n_ + [1] * n_ + [3] * n_
  num_labels = 4  # with blank
  # count_by_label = {i: len([x for x in input_seq if x == i]) for i in range(num_labels)}
  # assert len(set(count_by_label.values())) == 1  # all same
  opts = dict(
    loss_type="sum",
    target_seq=target_seq,
    #model_type="mem",
    #model_type={},  # 1-layer FFNN + bias
    #model_type={"bias": False},  # 1-layer FFNN - bias. note: this does not converge to peaky, depending on the example
    model_type={"layers": [{"class": "blstm", "dim": 500, "bias": True}], "bias": True},
    log_params=False, fast=True,
    num_frames=None,
    num_classes=num_labels - 1,  # except sil/blank
    input_seq=input_seq,
    init_type="rnd_uniform",  # important for BLSTM
    #init_type="rnd_normal",  # important for BLSTM
    #init_type="zero",
    #init_type="identity",
    rnd_scale=0.01,
    rnd_seed=1,
    #opt_class=tf.train.AdamOptimizer,
    learning_rate=0.1,
    #learning_rate=1.,
    num_steps=1000,
  )
  init_opts = opts.copy()
  init_opts.update(dict(num_steps=1, learning_rate=0, init_type="zero"))
  vs = run(**init_opts)
  key = "abwv"
  init_val = vs[key].tolist()

  tf.reset_default_graph()
  vs = run(**opts)
  final_val = vs[key].tolist()
  losses = vs["loss_values"]
  loss_below_thr = [(l < 1.) for l in losses].index(True)

  opts.update(dict(
    init_params=vs["pv"],
    num_steps=1, learning_rate=0, fast=False,
    input_seq=cv_input_seq, target_seq=cv_target_seq))
  tf.reset_default_graph()
  run(**opts)

  return init_val, final_val, loss_below_thr


def run_ratio_t_n():
  """
  Role of the ratio T/N.
  Simulation 6.1.
  """
  collected = {}
  for n in range(1, 10):
    a, b, l = _run_ratio_t_n(n)
    collected[n] = a, b, l
  print("-" * 40)
  pprint(collected)


def run_bias_sum():
  """
  Bias model with CTC, converges to peaky, even reinforced.
  Simulation 4.7.
  """
  opts = dict(
    loss_type="sum",
    model_type="bias",
    init_type="zero",
    num_frames=5,
    num_classes=1, target_seq=[0])
  run(**opts)


def run_ffnn_sum():
  """
  FFNN model with CTC, converges to peaky.
  Simulation 4.13.
  """
  n = 4
  opts = dict(
    loss_type="sum",
    model_type={"bias": False},
    num_frames=None,
    num_classes=1, target_seq=[0], input_seq=[1] * n + [0] * 2 * n + [1] * n,
    init_type="zero")
  run(**opts)


def run_mem_sum():
  """
  Memory model with CTC, converges to peaky, even reinforced.
  Simulation 4.16.
  """
  opts = dict(
    loss_type="sum",
    model_type="mem",
    init_type="zero",
    num_frames=100,
    num_classes=1, target_seq=[0])
  run(**opts)


def run_ffnn_sum_with_prior():
  """
  Section 7, loss with prior (L_hybrid) does not have peaky behavior.
  Simulation 7.5.
  """
  n = 4
  opts = dict(
    loss_type="sum_with_prior",
    model_type={"bias": False},
    num_frames=None,
    num_classes=1, target_seq=get_std_fsa_1label(),
    input_seq=[1] * n + [0] * 2 * n + [1] * n,
    init_type="zero")
  run(**opts)


def run_ffnn_sum_with_prior_sg():
  """
  Section 7, loss with prior (L_hybrid) does not have peaky behavior.
  Stop-gradient on prior.
  Remark 7.6.
  """
  n = 4
  opts = dict(
    loss_type="sum_with_prior_sg",
    model_type={"bias": False},
    num_frames=None,
    num_classes=1, target_seq=[0], input_seq=[1] * n + [0] * 2 * n + [1] * n,
    init_type="zero")
  run(**opts)


def run_generative_sum():
  """
  Section 7, loss with generative model (L_generative) does not have peaky behavior.
  Simulation 7.13.

  Note that the output can be a bit confusing,
  as the Baum-Welch stdout has a wrong normalization
  (that is a different computation than what is being used for the loss),
  and also the softmax stdout does not make sense.
  """
  n = 4
  opts = dict(
    loss_type="gen_sum", model_type="gen_model_free",
    num_frames=None,
    num_classes=1, target_seq=get_std_fsa_1label(),
    input_seq=[1] * n + [0] * 2 * n + [1] * n,
    init_type="zero")
  run(**opts)


def _get_func_name_from_stack(depth: int) -> str:
  depth += 1  # remove ourselves
  f = sys._getframe()
  for _ in range(depth):
    f = f.f_back
  return f.f_code.co_name


def plot_loss_grad_map(xlabel=None, ylabel=None, **kwargs):
  filename = None
  func_name = _get_func_name_from_stack(depth=1)
  print("Caller:", func_name)
  if func_name and func_name.startswith("run_"):
    filename = "../figures/%s.pdf" % func_name[len("run_"):]
    print("Will save under:", filename)

  xi, yi, lossi, gradxi, gradyi = get_loss_grad_map(**kwargs)

  xi2 = xi ** 2
  yi2 = yi ** 2
  x0 = numpy.min(xi2, axis=0)
  y0 = numpy.min(yi2, axis=1)
  idx0 = (numpy.argmin(x0), numpy.argmin(y0))
  loss0 = lossi[idx0]
  print("loss at x=%f,y=%f: %f" % (xi[idx0], yi[idx0], loss0))

  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from mpl_toolkits.mplot3d import Axes3D  # import to enable projection="3d"
  # streamplot

  plt.streamplot(
    xi, yi, gradxi, gradyi, density=2,
    start_points=[(xi[idx], yi[idx]) for idx in numpy.ndindex(xi.shape)])

  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  #surf = ax.plot_surface(xi, yi, lossi)

  # plt.contourf(xi, yi, lossi)

  extent = (numpy.min(xi), numpy.max(xi), numpy.min(yi), numpy.max(yi))
  print("loss range", numpy.min(lossi), numpy.max(lossi))
  eps = 1e-32

  lossi += -numpy.min(numpy.minimum(lossi, 0))

  # lossi = numpy.maximum(lossi, 0.)  # values below 0 are just due to numerical instability?
  # lossi = numpy.log(lossi + 1.)
  # lossi = numpy.log(lossi + eps)
  # lossi = numpy.log(lossi + 1.)
  # lossi = numpy.log(lossi + 1.)
  # norm = colors.PowerNorm(gamma=1.)
  # norm = colors.DivergingNorm(vmin=0, vmax=numpy.max(lossi), vcenter=1)
  # norm = colors.LogNorm(vmin=eps, vmax=loss0, clip=True)
  norm = None

  lossi = numpy.log(lossi + 1.)
  lossi = numpy.log(lossi + eps)

  # Supported interpolation:
  # 'none', 'nearest', 'bilinear', 'bicubic',
  # 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
  # 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
  # 'lanczos'.
  im = plt.imshow(
    lossi, origin="lower", extent=extent, norm=norm, interpolation="bicubic", cmap=None)

  # plt.colorbar(im)

  plt.grid()
  if xlabel:
    plt.gca().set_xlabel(xlabel)
  if ylabel:
    plt.gca().set_ylabel(ylabel)
  plt.tight_layout()

  if filename:
    print("Save figure as:", filename)
    plt.savefig(filename)
  else:
    plt.show()


def run_loss_map_bias_sum():
  plot_loss_grad_map(
    value_range=(-1., 1.),
    loss_type="sum", model_type="bias",
    num_classes=1, target_seq=[0], input_seq=[1] * _N + [0] * 2 * _N + [1] * _N)


def test_count_all_paths_with_label_seq_partly_dominated_inefficient():
  fsa = get_std_fsa_1label()  # same as target_seq=[0]
  n = 4
  opts = dict(
    loss_type="sum", model_type="model_free",
    num_classes=1, target_seq=fsa, input_seq=[1] * n + [0] * 2 * n + [1] * n)
  alpha = 0.5
  prob_dom = numpy.exp(alpha) / (numpy.exp(alpha) + numpy.exp(-alpha))
  print("prob dom:", prob_dom)
  _, grad0, bw0 = get_loss_grad_single(param_value=[0., 0.], **opts)
  _, grad1, bw1 = get_loss_grad_single(param_value=[-alpha, 0.], **opts)
  _, grad2, bw2 = get_loss_grad_single(param_value=[0., alpha], **opts)
  # print(bw1)
  # print(bw2)
  # print(grad1)
  # print(grad2)
  from fst_utils import count_all_paths_with_label_in_frame
  from fst_utils import count_all_paths_with_label_seq_partly_dominated_inefficient
  from fst_utils import Label1StrTemplate, BlankLabel, Label1
  num_frames = n * len(Label1StrTemplate)
  assert bw0.shape == bw1.shape == bw2.shape == (num_frames, 2)

  num_frames_sym, t_sym, c_a_sym = count_all_paths_with_label_in_frame(fsa=fsa, label=Label1)
  num_frames_sym_, t_sym_, c_b_sym = count_all_paths_with_label_in_frame(fsa=fsa, label=BlankLabel)
  for t in range(num_frames):
    c_a = int(c_a_sym.subs(num_frames_sym, num_frames).subs(t_sym, t).doit())
    c_b = int(c_b_sym.subs(num_frames_sym_, num_frames).subs(t_sym_, t).doit())
    z = c_a + c_b
    soft = [float(c_a) / z, float(c_b) / z]
    numpy.testing.assert_allclose(bw0[t], soft, rtol=1e-5)

  res_ = count_all_paths_with_label_seq_partly_dominated_inefficient(
    fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n,
    prob_dom=0.5, normalized=False, verbosity=1)
  for input_label in [Label1, BlankLabel]:
    c = 0
    res_by_label = numpy.zeros([2])
    c_a = c_b = 0
    for i, input_label_ in enumerate(Label1StrTemplate):
      if input_label_ != input_label:
        c += n
        for j in range(i * n, i * n + n):
          res_by_label += bw0[j]
          c_a += int(c_a_sym.subs(num_frames_sym, num_frames).subs(t_sym, j).doit())
          c_b += int(c_b_sym.subs(num_frames_sym_, num_frames).subs(t_sym_, j).doit())
    res_by_label /= c

    res_by_label_ = res_[(input_label, {Label1: BlankLabel, BlankLabel: Label1}[input_label])]
    assert c_a == res_by_label_[Label1] and c_b == res_by_label_[BlankLabel]
    res_by_label_ = numpy.array([res_by_label_[Label1], res_by_label_[BlankLabel]], dtype="float32")
    res_by_label_ /= sum(res_by_label_)
    numpy.testing.assert_allclose(res_by_label, res_by_label_)

  bws = {Label1: bw1, BlankLabel: bw2}
  res = {}
  for input_label in [Label1, BlankLabel]:
    c = 0
    res_by_label = {Label1: 0.0, BlankLabel: 0.0}
    for i, input_label_ in enumerate(Label1StrTemplate):
      if input_label_ != input_label:
        c += n
        for j in range(i * n, i * n + n):
          res_by_label[Label1] += bws[input_label][j][0]
          res_by_label[BlankLabel] += bws[input_label][j][1]
    res_by_label = {k: v / c for (k, v) in res_by_label.items()}
    res[(input_label, {Label1: BlankLabel, BlankLabel: Label1}[input_label])] = res_by_label
  print(res)
  res_ = count_all_paths_with_label_seq_partly_dominated_inefficient(
    fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n, prob_dom=prob_dom)
  print(res_)
  assert set(res.keys()) == set(res_.keys())
  for key, res_by_label_ in res_.items():
    res_by_label = res[key]
    assert set(res_by_label.keys()) == set(res_by_label_.keys()) == {Label1, BlankLabel}
    for label in [Label1, BlankLabel]:
      numpy.testing.assert_allclose(res_by_label[label], res_by_label_[label])


def run_loss_map_discriminative_sum():
  # n<=3 is not peaky. n>=4 is peaky.
  n = 10
  plot_loss_grad_map(
    value_range=(-5., 5.), steps=31,
    loss_type="sum", model_type="model_free",
    xlabel=r"$\theta_a$", ylabel=r"$\theta_b$",
    num_classes=1, target_seq=[0], input_seq=[1] * n + [0] * 2 * n + [1] * n)


def run_loss_map_discriminative_sum_with_prior():
  # n<=3 is not peaky. n>=4 is peaky.
  n = 10
  plot_loss_grad_map(
    value_range=(-5., 5.), steps=11,
    loss_type="sum_with_prior", model_type="model_free",
    xlabel=r"$\theta_a$", ylabel=r"$\theta_b$",
    num_classes=1, target_seq=get_std_fsa_1label(), input_seq=[1] * n + [0] * 2 * n + [1] * n)


def run_loss_map_discriminative_sum_with_stop_grad_prior():
  # n<=3 is not peaky. n>=4 is peaky.
  n = 10
  plot_loss_grad_map(
    value_range=(-5., 5.), steps=11,
    loss_type="sum_with_prior_sg", model_type="model_free",
    xlabel=r"$\theta_a$", ylabel=r"$\theta_b$",
    num_classes=1, target_seq=[0], input_seq=[1] * n + [0] * 2 * n + [1] * n)


def run_loss_map_generative_sum():
  # n<=3 is not peaky. n>=4 is peaky.
  n = 10
  plot_loss_grad_map(
    value_range=(-5., 5.), steps=21,
    loss_type="gen_sum", model_type="gen_model_free",
    xlabel=r"$\theta_a$", ylabel=r"$\theta_b$",
    num_classes=1, target_seq=get_std_fsa_1label(), input_seq=[1] * n + [0] * 2 * n + [1] * n)


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("command")
  args = arg_parser.parse_args()
  if args.command:
    globals()[args.command]()
    return
  # avg_blank_prob_hist(**DefaultOpts)
  run(**DefaultOpts)


if __name__ == "__main__":
  from returnn import better_exchook
  better_exchook.install()
  try:
    main()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
