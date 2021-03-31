#!/usr/bin/env python3
# vim: sw=2
"""
Implementation of the RNN-T loss in pure TF1,
plus comparisons against reference implementations.
"""
import os
import sys

import tensorflow.compat.v1 as tf
NEG_INF = -float("inf")


def py_print_iteration_info(msg, var, n, debug=True):
  """adds a tf.print op to the graph while ensuring it will run (when the output is used)."""
  if not debug:
    return var
  var_print = tf.print("n=", n, "\t", msg, tf.shape(var), var,
                       summarize=-1, output_stream=sys.stdout)
  with tf.control_dependencies([var_print]):
    var = tf.identity(var)
  return var


def backtrack_alignment_tf(bt_ta, input_lengths, label_lengths, blank_index):
  """Computes the alignment from the backtracking matrix.
  :param tf.TensorArray bt_ta: [T+U] * (B, U, 2)
  :param tf.Tensor input_lengths: (B,)
  :param tf.Tensor label_lengths: (B,)
  :param int blank_index:
  """
  max_path = bt_ta.size()
  n_batch = tf.shape(input_lengths)[0]
  alignments_ta = tf.TensorArray(
    dtype=tf.int32,
    size=max_path,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None,),  # [V]
    name="alignments"
  )
  initial_idx = tf.zeros((n_batch, 2), dtype=tf.int32)

  def body(s, alignments, idx):
    """Runs over s=[max_path-1,..., 1] (along the alignment)
    :param int s: path index
    :param tf.TensorArray alignments: [T+U] * (V,)
    :param tf.Tensor idx: (B, 2) -> [from_u, symbol/blank]
    """
    backtrack = bt_ta.read(s)  # (B, U, 2)

    init_u = tf.where(tf.greater_equal(s, input_lengths + label_lengths - 1),
                      label_lengths,  # if we are at the end of some path (or behind)
                      idx[:, 0]  # if we are within a path, continue backtracking.
                      )
    backtrack_indices = tf.stack([tf.range(n_batch), init_u], axis=-1)  # (B, 2)
    idx = tf.gather_nd(backtrack, backtrack_indices)
    align_write = tf.where(
      tf.less_equal(s, input_lengths + label_lengths),
      idx[:, 1],  # within alignment
      tf.ones((n_batch,), dtype=tf.int32) * blank_index)  # outside, assume blank
    alignments = alignments.write(s, align_write)
    return s-1, alignments, idx
  init_s = max_path-1
  final_s, final_alignments_ta, final_idx = tf.while_loop(lambda s, *args: tf.greater_equal(s, 1),
                                                          body, (init_s, alignments_ta, initial_idx))
  final_alignments_ta = final_alignments_ta.write(0, tf.tile([blank_index], [n_batch]))
  return tf.transpose(final_alignments_ta.stack())


def tf_shift_logprobs(mat, axis):
  """
  Shifts the log-probs per-batch row-wise.

  :param mat: (B, U, T, V)
  :param axis:
  :return: (B, T+U+1, U, V)
  """
  # mat: (B, T, U, V)
  # axis_to_expand: usually U
  # axis: usually T
  # batch-axis has to be first
  max_time = tf.shape(mat)[axis]  # T

  def fn(args):  # x: (B, U, V)
    """Computes the shift per diagonal and pads accordingly."""
    x, shift = args
    padded = tf.pad(x, [[0, 0],  # B
                        [shift, max_time - shift],  # U+T+1
                        [0, 0]  # V
                        ], constant_values=0)
    return padded, shift

  elems0 = tf.transpose(mat, [1, 0, 2, 3])  # [T, B, U, V]
  elems1 = tf.range(max_time)  # [T]
  t, _ = tf.map_fn(fn, elems=(elems0, elems1))  # T* [B, T+U+1, V]
  t = tf.transpose(t, [1, 0, 2, 3])  # [B, T, U+1, V]
  return t


def rnnt_loss(log_probs, labels, input_lengths=None, label_lengths=None,
              blank_index=0, debug=False, with_alignment=False):
  """
  Computes the batched forward pass of the RNN-T model.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param tf.Tensor log_probs: (B, T, U+1, V) log-probabilities
  :param tf.Tensor labels: (B, U) -> [V] labels
  :param tf.Tensor input_lengths: (B,) length of input frames
  :param tf.Tensor label_lengths: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool debug: enable verbose logging
  :param bool with_alignment: whether to generate the alignments or not.
  :return:
  with_alignment=True -> (costs, alignments)
                =False -> costs
  """
  """Pure TF implementation of the RNN-T loss."""
  shape = tf.shape(log_probs)
  n_batch = shape[0]     # B
  max_time = shape[1]    # T
  max_target = shape[2]  # U

  log_probs_tr = tf.transpose(log_probs, [0, 2, 1, 3])  # (B, T, U, V) -> (B, U, T, V)
  log_probs_shifted = tf_shift_logprobs(log_probs_tr, axis=1)  # (B, U+T+1, U, V)

  num_diagonals = max_time + max_target

  labels = py_print_iteration_info("labels", labels, 0, debug=debug)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, None),  # (B, U, V)
    name="log_probs_shifted",
  )
  # (B, U+T+1, U, V) -> [(B, U, V)] * (U+T+1)
  log_probs_ta = log_probs_ta.unstack(tf.transpose(log_probs_shifted, [2, 0, 1, 3]))

  init_alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None,),  # (B, n)
    name="alpha_diagonals",
  )
  init_alpha_ta = init_alpha_ta.write(1, tf.zeros((n_batch, 1)))

  if with_alignment:
    # backtrack matrix, for each diagonal and target -> [from_u, symbol/blank]
    init_backtrack_ta = tf.TensorArray(
      dtype=tf.int32,
      size=num_diagonals-1,
      dynamic_size=False,
      infer_shape=False,
      element_shape=(None, None, 2),
      name="backtrack"
    )
  else:
    init_backtrack_ta = None

  def cond(n, *_):
    """We run the loop until all elements are covered by diagonals.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta, *args):
    """body of the while_loop, loops over the diagonals of the alpha-tensor."""
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    lp_diagonal = log_probs_ta.read(n-2)[:, :n-1, :]  # (B, U|n, V)
    lp_diagonal = py_print_iteration_info("lp_diagonal", lp_diagonal, n, debug=debug)

    prev_diagonal = alpha_ta.read(n-1)[:, :n]  # (B, n-1)
    prev_diagonal = py_print_iteration_info("prev_diagonal", prev_diagonal, n, debug=debug)

    alpha_blank = prev_diagonal  # (B, N)
    alpha_blank = tf.concat([alpha_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)

    # (B, U, V) -> (B, U)
    lp_blank = lp_diagonal[:, :, blank_index]  # (B, U)
    lp_blank = tf.concat([lp_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    lp_blank = py_print_iteration_info("lp(blank)", lp_blank, n, debug=debug)

    # (B,N-1) ; (B,1) ->  (B, N)
    alpha_y = prev_diagonal
    alpha_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), alpha_y], axis=1)
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    labels_max_len = tf.minimum(max_target-1, n-1)
    labels_shifted = labels[:, :labels_max_len]  # (B, U-1|n-1)
    labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    batchs, rows = tf.meshgrid(
      tf.range(n_batch),
      tf.range(labels_max_len),
      indexing='ij'
    )
    lp_y_indices = tf.stack([batchs, rows, labels_shifted], axis=-1)  # (B, U-1|n-1, 3)
    lp_y_indices = py_print_iteration_info("lp_y_indices", lp_y_indices, n, debug=debug)
    lp_y = tf.gather_nd(lp_diagonal[:, :, :], lp_y_indices)  # (B, U)
    # (B, U) ; (B, 1) -> (B, U+1)
    lp_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), lp_y], axis=1)
    lp_y = py_print_iteration_info("lp(y)", lp_y, n, debug=debug)

    cut_off = max_target
    alpha_y = tf.cond(tf.greater(n, max_target),
                      lambda: alpha_y[:, :cut_off],
                      lambda: alpha_y)
    lp_blank = tf.cond(tf.greater(n, max_target),
                       lambda: lp_blank[:, :cut_off],
                       lambda: lp_blank)
    alpha_blank = tf.cond(tf.greater(n, max_target),
                          lambda: alpha_blank[:, :cut_off],
                          lambda: alpha_blank)

    # all should have shape (B, n)
    blank = alpha_blank + lp_blank
    y = alpha_y + lp_y
    red_op = tf.stack([blank, y], axis=0)  # (2, B, N)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_diagonal = tf.math.reduce_logsumexp(red_op, axis=0)  # (B, N)

    new_diagonal = new_diagonal[:, :n]
    new_diagonal = py_print_iteration_info("new_diagonal", new_diagonal, n, debug=debug)

    if with_alignment:
      backtrack_ta = args[0]
      argmax_idx = tf.argmax([blank, y], axis=0)
      max_len_diag = tf.minimum(n, max_target)
      u_ranged = tf.tile(tf.range(max_len_diag)[None], [n_batch, 1])  # (B, n|U)
      blank_tiled = tf.tile([[blank_index]], [n_batch, 1])

      stack_blank_sel = tf.stack([u_ranged, tf.tile(blank_tiled, [1, max_len_diag])], axis=-1)

      b, r = tf.meshgrid(
        tf.range(n_batch),
        tf.maximum(0, tf.range(max_len_diag) - 1),
        indexing='ij'
      )
      labels_indices = tf.stack([b, r], axis=-1)
      labels_emit_sel = tf.gather_nd(labels, labels_indices)  # (B, n)  labels[u-1]
      stack_emit_sel = tf.stack([u_ranged-1, labels_emit_sel], axis=-1)
      best_sel = tf.where(tf.tile(tf.equal(argmax_idx, 0)[..., None], [1, 1, 2]),
                          stack_blank_sel,  # blank
                          stack_emit_sel  # emit
                          )
      backtrack_ta = backtrack_ta.write(n-1, best_sel)
    else:
      backtrack_ta = None
    return [n + 1, alpha_ta.write(n, new_diagonal)] + ([backtrack_ta] if with_alignment else [])

  init_n = tf.constant(2)
  if with_alignment:
    final_n, alpha_out_ta, backtrack_out_ta = tf.while_loop(cond, body_forward,
                                                            [init_n, init_alpha_ta, init_backtrack_ta],
                                                            parallel_iterations=1,  # iterative computation
                                                            name="rnnt")
  else:
    final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [init_n, init_alpha_ta],
                                          parallel_iterations=1,  # iterative computation
                                          name="rnnt")
    backtrack_out_ta = None

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)

  # (B,): batch index -> diagonal index
  diag_idxs = input_lengths + label_lengths  # (B,)

  # (B,): batch index -> index within diagonal
  within_diag_idx = label_lengths

  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=True,
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_diagonals",
  )

  def ta_read_body(i, res_loop_ta):
    """Reads from the alpha-diagonals TensorArray. We need this because of the inconsistent shapes in the TA."""
    ta_item = alpha_out_ta.read(diag_idxs[i])[i]
    return i+1, res_loop_ta.write(i, ta_item[within_diag_idx[i]])

  final_i, a_ta = tf.while_loop(
    lambda i, _: tf.less(i, n_batch),
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  indices = tf.stack([
    tf.range(n_batch),
    input_lengths-1,  # noqa T-1
    label_lengths,    # U-1
    tf.tile([blank_index], [n_batch]),
  ], axis=-1)  # (B, 3)
  ll_tf = a_ta.stack() + tf.gather_nd(log_probs, indices)

  if with_alignment:
    assert backtrack_out_ta is not None
    alignments = backtrack_alignment_tf(backtrack_out_ta, input_lengths, label_lengths, blank_index)
    return ll_tf, alignments
  else:
    return ll_tf
