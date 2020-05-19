#!/usr/bin/env python3
# vim: sw=2
"""
Implementation of the RNA loss in pure TF,
plus comparisons against reference implementations.
This is very similar to RNN-T loss, but restricts
the paths to be strictly monotonic.

references:
  * recurrent neural aligner:
      https://pdfs.semanticscholar.org/7703/a2c5468ecbee5b62c048339a03358ed5fe19.pdf
"""
import os
import sys
import numpy as np
import tensorflow as tf
from termcolor import colored
from ref_transduce import transduce as transduce_ref
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "returnn"))
from ref_rna import forward_pass, analytical_gradient, backward_pass, numerical_gradient

NEG_INF = -float("inf")


def logsumexp(*args):  # summation in linear space -> LSE in log-space
  """
  Stable log sum exp.
  """
  if all(a == NEG_INF for a in args):
    return NEG_INF
  a_max = max(args)
  lsp = np.log(sum(np.exp(a - a_max)
                   for a in args))
  return a_max + lsp


def log_softmax(acts, axis=None):
  """computes log(softmax(x, axis)) in a numerical stable way."""
  assert axis is not None
  a = acts - np.max(acts, axis=axis, keepdims=True)  # normalize for stability
  probs = np.sum(np.exp(a), axis=axis, keepdims=True)
  log_probs = a - np.log(probs)
  return log_probs


def py_print_iteration_info(msg, var, n, *vars, debug=True):
  """adds a tf.print op to the graph while ensuring it will run (when the output is used)."""
  if not debug:
    return var
  var_print = tf.print("n=", n, "\t", msg, tf.shape(var), var, *vars,
                       summarize=-1, output_stream=sys.stdout)
  with tf.control_dependencies([var_print]):
    var = tf.identity(var)
  return var


def numpy_forward_naive(log_probs, labels, blank_index, label_rep=False, with_alignment=False, debug=False):
  """Forward calculation of the RNA loss."""
  n_time, n_target, n_vocab = log_probs.shape  # (T, U+1, V)
  alpha = np.zeros((n_time+1, n_target))  # 1 in log-space
  # print("a = alpha[t-1, u - 1] + log_probs[t - 1, u - 1, labels[u - 1]]")
  # print("b = alpha[t-1, u] + log_probs[t-1, u,  blank]")
  if debug:
    print("labels: ", labels)
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
  bt_mat = np.ones((n_time+1, n_target+1, 2), dtype=np.int32) * 0  # store hat{x_j} of the most likely path so far
  # bt_label =
  # bt_mat_label = np.zeros((n_time + 1, n_target), dtype=np.int32)  # store hat{y_j} of the current most likely path
  # blank_state = max(n_target - 1, n_time)  # so that it is definitely last
  # blank_state = n_target - 1
  blank_state = 0
  bt_mat[0, 0] = (0, 0)

  for t in range(1, n_time+1):
    # blank - blank - blank - ...
    alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank_index]
    bt_mat[t, 0] = (0, blank_index)
    if debug:
      print('t=%2d u= 0: alpha[%d, 0] + log_probs[%d, 0, %d] = %.3f + %.3f = %.3f' % (
        t, t-1, t-1, blank_index, alpha[t - 1, 0], log_probs[t - 1, 0, blank_index], alpha[t, 0]))

    # label - label - label - ...
    if t < n_target:
      alpha[t, t] = alpha[t-1, t-1] + log_probs[t-1, t-1, labels[t-1]]
      bt_mat[t, t] = (t-1, labels[t-1])
      # bt_mat_label[t-1, t-1] = labels[t-1]
      # bt_mat_score[t, t] = log_probs[t-1, t-1, labels[t-1]]

    for u in range(1, min(t, n_target)):
      skip = alpha[t - 1, u] + log_probs[t - 1, u, blank_index]
      emit = alpha[t - 1, u - 1] + log_probs[t - 1, u - 1, labels[u - 1]]
      sum_args = [skip, emit]
      # output label repetition allowed:
      # see figure https://github.com/1ytic/warp-rna/blob/master/aligner.gif
      # we allow the 'd' symbol to be output, from the arc (T=2,U=1) -> (T=3, U=1)
      # but we need to take care to not include the arc for the first output (emit)
      if label_rep and t-u > 1:  # enabled and not on the diagonal line
        same = alpha[t-1, u] + log_probs[t-1, u, labels[u-1]]
        if debug:
          print("t=%2d u=%2d same=%.3f a=%.3f lp[%d, %d, labels[%d]]=%3f label=%s" %
                (t, u, same, alpha[t-1, u], t-1, u, u-1, log_probs[t-1, u, labels[u-1]], labels[u-1]))
        sum_args += [same]
      max_prob_idx = int(np.argmax(sum_args))
      # list-idx -> state-idx
      argmax_dict = {0: (u, blank_index),  # skip, blank symbol.
                     1: (u-1, labels[u-1]),  # (t-1, u-1),  # emit
                     2: (u, labels[u-1]),    # (t-1, u),    # same
                     }
      argmax_tuple = argmax_dict[max_prob_idx]
      if debug or True:
        print("[np naive] BT[%2d,%2d] = (%s) %d state=%d, label=%d" % (
          t - 1, u, {0: "skip", 1: "emit", 2: "same"}[max_prob_idx],
          max_prob_idx, argmax_tuple[0], argmax_tuple[1]))
      bt_mat[t, u] = argmax_tuple

      alpha[t, u] = elem = logsumexp(*sum_args)  # addition in linear-space -> LSE in log-space
      if debug:
        print('t=%2d u=%2d: LSE(%.3f + %.3f, %.3f +  %.3f) = LSE(%.3f, %.3f) = %.3f' % (t, u,
                                                                                        alpha[t - 1, u],
                                                                                        log_probs[t-1, u, blank_index],
                                                                                        alpha[t - 1, u - 1],
                                                                                        log_probs[t - 1, u - 1,
                                                                                                  labels[u - 1]],
                                                                                        skip, emit, elem))

  if debug:
    assert len(alpha.shape) == 2
    print("Alpha matrix: (%d, %d)" % tuple(alpha.shape))
    np.set_printoptions(precision=3, linewidth=120)
    print(alpha)
  nll = - alpha[n_time, n_target-1]
  np.set_printoptions(precision=3, linewidth=120)
  # print("[np naive] backtrack matrix:")
  # print(bt_mat)
  # print("[np naive] backtrack matrix labels:")
  # print(bt_mat_label)
  alignment = compute_alignment(bt_mat, n_time, n_target)
  alignment_batched = compute_alignment_numpy_batched(bt_mat[np.newaxis, ...],
                                                      np.array([n_time]), np.array([n_target]))
  print("[np naive] batched alignment:", alignment_batched[0])
  with sess.as_default():
    align = compute_alignment_tf(tf.expand_dims(bt_mat, axis=0),
                                 tf.constant([n_time]), tf.constant([n_target]))
    align_tf = align.eval()
    print("[np naive] TF alignment:", align_tf[0])
    np.testing.assert_equal(align_tf, alignment_batched)
  assert len(alignment) == n_time
  if debug:
    print("negative log-likelihood = - alpha[%d, %d] = %.4f" %
          (n_time, n_target-1, nll))
  if with_alignment:
    return alpha, nll, alignment
  else:
    return alpha, nll


def compute_alignment_numpy_batched(bt_mat, input_lens, label_lens):
  """Computes the alignment from the backtracking matrix.
  We do this in a batched fashion so we can compare/copy this directly to TF.

  :param bt_mat: backtracking matrix (B, T+1, U, 2)
  :param input_lens: (B,)
  :param label_lens: (B,)

  :return alignment of form (B, T) -> [V]
  :rtype np.ndarray
  """
  assert input_lens.shape == label_lens.shape
  n_batch, max_time, max_target, track_dim = bt_mat.shape
  assert track_dim == 2
  # blank_state = max_target - 1
  # (B, U) -> (B, U+1), add blank last state (such that we can do labels[idx])
  # labels = np.concatenate([labels, np.tile([[blank_index]], [n_batch])], axis=1)
  # labels = np.concatenate([np.tile([[blank_index]], [n_batch]), labels], axis=1)
  alignments = np.zeros((n_batch, max_time-1), dtype=np.int32)
  label_align = np.zeros_like(alignments)
  idx = bt_mat[np.arange(n_batch), input_lens, label_lens-1]
  initial_idx = idx
  for t in reversed(range(max_time-1)):
    assert idx.shape == (n_batch, 2)
    print("[np batched] t=%2d idx=" % t, idx)
    alignments[:, t] = np.where(t <= input_lens - 1, idx[:, 0], 0)  # (B,) masked invalid
    label_align[:, t] = np.where(t <= input_lens - 1, idx[:, 1], 0)  # (B,) masked invalid
    idx = bt_mat[np.arange(n_batch), np.array([t]), idx[:, 0]]  # (B,)
    cond = (t > input_lens - 1)[:, np.newaxis]
    idx = np.where(cond, initial_idx, idx)
    assert idx.shape == (n_batch, 2)
  # print(label_align)
  # return alignments
  return label_align


def compute_alignment_tf(bt_mat, input_lens, label_lens):
  """Computes the alignment from the backtracking matrix.
  We do this in a batched fashion so we can compare/copy this directly to TF.

  :param bt_mat: backtracking matrix (B, T+1, U, 2) where (2,) is (state-idx, label-idx)
  :param input_lens: (B,)
  :param label_lens: (B,)

  :return alignment of form (B, T) -> [V]
  :rtype np.ndarray
  """
  shape = tf.shape(bt_mat)
  n_batch, max_time, max_target = shape[0], shape[1], shape[2]

  alignments = tf.TensorArray(
    dtype=tf.int32,
    clear_after_read=False,
    size=max_time-1,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None,),  # (B,)
    name="alignments",
  )

  # (T, B) TA which holds the max-prob label sequence
  # this can prob be implemented more efficiently.
  label_align = tf.TensorArray(
    dtype=tf.int32,
    clear_after_read=False,
    size=max_time-1,
    dynamic_size=False,
    infer_shape=True,
    element_shape=(None,),  # (B,)
    name="label_align",
  )

  idxs = tf.stack([
    tf.range(n_batch),
    input_lens,
    label_lens-1
  ], axis=-1)  # (B, 3)
  initial_idx = tf.gather_nd(bt_mat, idxs)  # (B, 2)

  def body(t, alignments, label_align, idx):
    # backtracking state sequence
    alignments = alignments.write(t, tf.where(tf.less_equal(t, input_lens-1), idx[:, 0], tf.zeros_like(idx[:, 0])))  # (B,)
    label_align = label_align.write(t, tf.where(tf.less_equal(t, input_lens-1), idx[:, 1], tf.zeros_like(idx[:, 1])))
    idxs = tf.stack([
      tf.range(n_batch),  # (B,1)
      tf.tile([t], [n_batch]),  # (1,)
      idx[:, 0],  # (B,)
    ], axis=-1)
    idx = tf.gather_nd(bt_mat, idxs)
    idx = tf.where(tf.greater(t, input_lens - 1), initial_idx, idx)
    return t-1, alignments, label_align, idx
  t = max_time-2
  final_t, final_alignments, final_label_align_ta, final_idx = tf.while_loop(lambda t, idx, _, _2: tf.greater_equal(t, 0),
                                                       body,
                                                       (t, alignments, label_align, initial_idx))
  # final_alignments = final_alignments.write(0, final_idx)
  # label_idxs = tf.stack([
  #   tf.range(n_batch),  # (B,)
  #   final_idx,  # (B,)
  # ], axis=-1)  # (B, 2)
  # final_label_align_ta = final_label_align_ta.write(0, tf.where(tf.less_equal(t, input_lens - 1), tf.gather_nd(labels, label_idxs),
  #                                             tf.zeros_like(final_idx)))

  final_label_alignment = final_label_align_ta.stack()
  # return tf.transpose(final_alignments.stack())  # (T, B) -> (B, T)
  return tf.transpose(final_label_alignment)


def compute_alignment(bt_mat, input_len, label_len):
  """Computes the alignment from the backtracking matrix."""
  # n_target = labels.shape[0] + 1
  # labels = np.concatenate([[0], labels])
  # print("Computing alignment for T=%d, U-1=%d, label-seq: %s" %
  #       (input_len, label_len-1, labels))
  idx = bt_mat[input_len, label_len-1]
  # label = bt_mat_label[i, input_lens[i], label_lens[i]-1]
  alignment = np.ones((input_len,), dtype=np.int32) * 99
  # def map_idx_to_label(idx, labels, blank_index):
  #   return labels[idx]
    # return labels[idx] if idx != n_target else blank_index
  label_align = np.zeros_like(alignment)
  for t in reversed(range(1, input_len)):
    alignment[t] = idx[0]
    # last state == blank state
    label_align[t] = idx[1]
    print("align [t=%2d] state-idx=%2d, label=%2d" % (t, idx[0], label_align[t]))
    # print("(t=%2d, u=%2d) -> (t=%2d, u=%2d)" % (t-1, bt_mat[i, t, idx], t, idx))
    idx = bt_mat[t, idx[0]]
  alignment[0] = idx[0]
  label_align[0] = idx[1]
  # print(label_align)
  # print("align [t=%2d] state-idx=%2d" % (0, idx))
  return label_align


def numpy_forward_batched(log_probs, labels, blank_index, input_lens, label_lens, debug=False,
                          label_rep=False, with_alignment=False):
  """Forward calculation of the RNA loss using the same strategy as the TF impl."""
  n_batch, max_time, max_target, n_vocab = log_probs.shape  # (B, T, U, V)
  assert labels.shape == (n_batch, max_target-1)  # (B, U-1)
  # blank_state = max_target-1  # this is actually U
  if debug:
    print("U=%d, T=%d, V=%d" % (max_target-1, max_time, n_vocab))
    print("log-probs: (B=%d, T=%d, U+1=%d, V=%d)" % (n_batch, max_time, max_target, n_vocab))
    print("labels: (B=%d, U-1=%d)" % (n_batch, labels.shape[1]))
    print("seq-lengths: T=%r U=%r" % (input_lens, label_lens))

  def print_debug(n, *vars):
    """Some basic debug information printing."""
    if debug:
      print("[n=%2d]" % n, *vars)
  # alpha columns
  alphas = [[], np.zeros((n_batch, 1))]
  bt_mat = np.zeros((n_batch, max_time+1, max_target, 2), dtype=np.int32)

  for n in range(2, max_time+2):
    # actually previous one.
    lp_column = log_probs[:, n-2, :n-1]
    print_debug(n, "lp_column", lp_column)

    col_maxlen = min(max_target, n)
    prev_column = alphas[n-1][:, :col_maxlen]
    print_debug(n, "prev_column", prev_column)
    # skip = alpha[t - 1, u] + log_probs[t - 1, u, blank_index]
    alpha_blank = prev_column  # (B, N)
    alpha_blank = np.concatenate([alpha_blank, np.tile([[NEG_INF]], [n_batch, 1])], axis=1)

    # (B, U, V) -> (B, U)
    lp_blank = lp_column[:, :, blank_index]  # (B, U)
    lp_blank = np.concatenate([lp_blank, np.tile([[NEG_INF]], [n_batch, 1])], axis=1)

    # emit = alpha[t - 1, u - 1] + log_probs[t - 1, u - 1, labels[u - 1]]
    alpha_y = prev_column
    alpha_y = np.concatenate([np.tile([[NEG_INF]], [n_batch, 1]), alpha_y], axis=1)

    # NOTE:
    # We cut off the columns in the top-right corner,
    # as soon as we can make sure there are no interesting values.
    # this happens when n > U. This makes sure all values have the same shape.
    cut_off = max_target
    if n > cut_off:  # phase (c), cut off top-right
      alpha_y = alpha_y[:, :cut_off]
      lp_blank = lp_blank[:, :cut_off]
      alpha_blank = alpha_blank[:, :cut_off]

    labels_maxlen = min(max_target - 1, n - 1)
    labels_shifted = labels[:, :labels_maxlen]
    print_debug(n, "labels_shifted", labels_shifted)
    batchs_idxs, rows_idxs = np.meshgrid(
      np.arange(np.shape(labels_shifted)[0]),  # B
      np.arange(np.shape(labels_shifted)[1]),  # U-1
      indexing='ij'
    )
    lp_y = lp_column[batchs_idxs, rows_idxs, labels_shifted]
    lp_y = np.concatenate([np.tile([[NEG_INF]], [n_batch, 1]), lp_y], axis=1)

    # all should have shape (B, n)
    print_debug(n, colored("lp_blank", "green"), lp_blank)
    print_debug(n, colored("alpha_blank", "green"), alpha_blank)
    blank = alpha_blank + lp_blank
    print_debug(n, colored("blank", "green"), blank)

    print_debug(n, "labels_shifted", labels_shifted)
    print_debug(n, colored("lp_y", "red"), lp_y)
    print_debug(n, colored("alpha_y", "red"), alpha_y)
    y = alpha_y + lp_y
    print_debug(n, colored("y", "red"), y)
    sum_args = [blank, y]

    new_column = np.logaddexp(blank, y)  # (B, N)

    # label repetition allowed, see naive numpy impl for explanation
    if label_rep and n > 3:
      col_len = np.minimum(n-1, max_target)
      # not first and not on diagonal
      mask = np.logical_and(np.arange(col_len) > 0, np.arange(col_len) < n-2)  # (n-1,)
      mask_exp = np.expand_dims(mask, axis=0)  # (1, n-1)
      print_debug(n, colored("mask", "yellow"), mask)
      alpha_same = np.where(np.tile(mask_exp, [n_batch, 1]), prev_column, NEG_INF)
      alpha_same = np.concatenate([alpha_same, np.tile([[NEG_INF]], [n_batch, 1])], axis=1)
      if n > cut_off:  # phase (c), cut off top-right
        alpha_same = alpha_same[:, :cut_off]
      assert alpha_same.shape == alpha_blank.shape
      print_debug(n, colored("alpha_same", "yellow"), alpha_same)

      # labels_maxlen = min(max_target - 1, n - 1)
      labels_maxlen_same = min(max_target - 1, n-3)
      batchs_idxs, rows_idxs = np.meshgrid(
        np.arange(n_batch),  # B
        np.arange(labels_maxlen_same)+1,  # U-1
        # np.arange(labels_maxlen_same),
        indexing='ij'
      )
      # from (B, U, V) gather (B, N) values
      lp_same = lp_column[batchs_idxs, rows_idxs, labels[:, :labels_maxlen_same]]
      # pad the values so we can add the scores
      # num_pads = min(2, n - labels_maxlen_same) # min(1, max_target - n + 2)
      # print("num_pads", num_pads)
      lp_same = np.concatenate([np.tile([[NEG_INF]], [n_batch, 1]),
                                lp_same,
                                np.tile([[NEG_INF]], [n_batch, 2])], axis=1)
      if n > cut_off:  # phase (c), cut off top-right
        lp_same = lp_same[:, :cut_off]
      print_debug(n, colored("lp_same", "yellow"), lp_same)
      same = alpha_same + lp_same
      sum_args += [same]
      new_column = np.logaddexp(new_column, same)

    # sum_args: (3|2, B, N)
    # we want to compute this:
    # argmax_dict = {0: (u, blank_index),  # skip, blank symbol.
    #                1: (u - 1, labels[u - 1]),  # (t-1, u-1),  # emit
    #                2: (u, labels[u - 1]),  # (t-1, u),    # same
    #                }
    # bt_mat[t, u] = argmax_dict[argmax_idx]

    argmax_idx = np.argmax(sum_args, axis=0)  # (B, N) -> [2|3]
    max_len = sum_args[0].shape[1]  # np.minimum(n-1, max_target)
    u_ranged = np.tile(np.expand_dims(np.arange(max_len), axis=0), [n_batch, 1])  # (B, U|n)
    # (B, U) ; (B, 1) -> (B, U+1), we do this to allow vectorized access to the labels
    blank_tiled = np.tile([[blank_index]], [n_batch, 1])  # (B, 1)
    labels_exp = np.concatenate([labels, blank_tiled], axis=1)
    u_ranged_shifted = u_ranged - 1
    # np.meshgrid(
    #
    # )
    b, r = np.meshgrid(
      np.arange(n_batch),
      np.maximum(0, np.arange(max_len) - 1),
      indexing='ij'
    )
    label_idxs = np.stack([b, r], axis=-1)
    # labels_emit = labels_exp[label_idxs]  # (B, n)  labels[u-1]
    # labels_emit = labels_exp[np.arange(n_batch), np.maximum(0, np.arange(max_len) - 1)]
    labels_emit = labels_exp[np.arange(n_batch)[:, np.newaxis], np.maximum(0, np.arange(max_len)[np.newaxis, :]-1)]
    labels_same = labels_emit

    # labels_emit = labels_exp[np.arange(n_batch), u_ranged_shifted]  # (B, n) labels[u-1]
    # labels_same = labels_exp[np.arange(n_batch), u_ranged_shifted]  # labels[u-1]
    # we track the state where the arc came from:
    # bt_mat: (B, T, U, 2)           blank           emit           same
    # last dimension is (state-idx, label-idx)
    sel_blank = np.stack([u_ranged, np.tile(blank_tiled, [1, max_len])], axis=-1)  # (B,)
    sel_emit = np.stack([u_ranged_shifted, labels_emit], axis=-1)  # (1,U|n) | (B, U|n)-> (B, n, 2)
    # bt_mat[np.arange(n_batch), np.array([t]), idx[:, 0]]  # (B,)
    sel_same = np.stack([u_ranged, labels_same], axis=-1)
    sel = np.where((argmax_idx == 0)[...,np.newaxis],
                   sel_blank,  # blank
                   np.where((argmax_idx == 1)[...,np.newaxis],
                            sel_emit,  # emit
                            sel_same))  # same

    bt_mat[:, n-1, :max_len] = sel

    print_debug(n, "new_column", new_column)
    alphas.append(new_column)  # s.t. alphas[n] == new_column

    if debug:
      print("\n")

  list_nll = []
  col_idxs = input_lens + 1  # (B,)
  if debug:
    print("[np batched] T=%r U=%r" % (input_lens, label_lens))

  # backtracking
  alignments_np = compute_alignment_numpy_batched(bt_mat, input_lens, label_lens+1)

  # (B,): batch index -> index within column
  # We need to handle the U>T case for each example.
  within_col_idx = label_lens
  for i in range(n_batch):
    ta_item = alphas[col_idxs[i]]  # (B, N)

    a = ta_item[i, within_col_idx[i]]
    # b = log_probs[i, input_lens[i]-1, label_lens[i], blank_index]
    if debug:
      print("FINAL i=%d, col_idx=%d, within_col_idx=%d, col=%r" % (i, col_idxs[i], within_col_idx[i], ta_item[i]))
      print("FINAL i=%d" % i, "NLL=%.3f" % (-a))
    nll = -a  # + b
    list_nll.append(nll)
  if with_alignment:
    return np.array(list_nll), alignments_np  # (B,)
  else:
    return np.array(list_nll)  # (B,)


def tf_forward_shifted_rna(log_probs, labels, input_lengths=None, label_lengths=None, blank_index=0,
                           label_rep=False, with_alignment=False, debug=False):
  """
  Computes the batched forward pass of the RNA model.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param tf.Tensor log_probs: (B, T, U+1, V) log-probabilities
  :param tf.Tensor labels: (B, U) -> [V] labels
  :param tf.Tensor input_lengths: (B,) length of input frames
  :param tf.Tensor label_lengths: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool with_alignment: Also computes and returns the best-path alignment
  :param bool debug: enable verbose logging
  :return:
  """
  shape = tf.shape(log_probs)
  n_batch = shape[0]     # B
  max_time = shape[1]    # T
  max_target = shape[2]  # U+1

  blank_state = max_target - 1  # this is actually U

  num_columns = max_time + 2

  labels = py_print_iteration_info("labels", labels, 0, debug=debug)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_columns,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, None),  # (B, U, V)
    name="log_probs",
  )
  # (B, T, U, V) -> [(B, U, V)] * (T)
  log_probs_ta = log_probs_ta.unstack(tf.transpose(log_probs, [1, 0, 2, 3]))

  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_columns,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None,),  # (B, n)
    name="alpha_columns",
  )
  alpha_ta = alpha_ta.write(1, tf.zeros((n_batch, 1)))

  if with_alignment:
    bt_ta = tf.TensorArray(
      dtype=tf.int32,
      clear_after_read=False,
      size=num_columns-1,
      dynamic_size=False,
      infer_shape=True,
      element_shape=(None, None, 2),  # (B, U)
      name="bt_columns",
    )  # (T, B, U)
    zero_state = tf.zeros((n_batch, max_target), dtype=tf.int32)
    initial_align_tuple = tf.stack([zero_state, tf.ones_like(zero_state)*blank_index], axis=-1)  # (B, U, 2)
    bt_ta = bt_ta.write(0, initial_align_tuple)
  else:
    bt_ta = None

  def cond(n, *args):
    """We run the loop until all input-frames have been consumed.
    """
    return tf.less(n, num_columns)

  def body_forward(n, alpha_ta, *args):
    """body of the while_loop, loops over the columns of the alpha-tensor."""
    # alpha(t-1,u-1) + logprobs(t-1, u-1)
    # alpha_blank      + lp_blank

    lp_column = log_probs_ta.read(n-2)[:, :n-1, :]  # (B, U|n, V)
    lp_column = py_print_iteration_info("lp_column", lp_column, n, debug=debug)

    column_maxlen = tf.reduce_min([max_target, n])
    prev_column = alpha_ta.read(n-1)[:, :column_maxlen]  # (B, n-1)
    prev_column = py_print_iteration_info("prev_column", prev_column, n, debug=debug)

    alpha_blank = prev_column  # (B, N)
    alpha_blank = tf.concat([alpha_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)

    # (B, U, V) -> (B, U)
    lp_blank = lp_column[:, :, blank_index]  # (B, U)
    lp_blank = tf.concat([lp_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    lp_blank = py_print_iteration_info("lp(blank)", lp_blank, n, debug=debug)

    # (B,N-1) ; (B,1) ->  (B, N)
    alpha_y = prev_column
    alpha_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), alpha_y], axis=1)
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    labels_maxlen = tf.minimum(max_target-1, n-1)
    labels_shifted = labels[:, :labels_maxlen]  # (B, U-1|n-1)
    labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    batchs, rows = tf.meshgrid(
      tf.range(n_batch),
      tf.range(labels_maxlen),
      indexing='ij'
    )
    lp_y_idxs = tf.stack([batchs, rows, labels_shifted], axis=-1)  # (B, U-1|n-1, 3)
    lp_y_idxs = py_print_iteration_info("lp_y_idxs", lp_y_idxs, n, debug=debug)
    lp_y = tf.gather_nd(lp_column[:, :, :], lp_y_idxs)  # (B, U)
    # (B, U) ; (B, 1) -> (B, U+1)
    lp_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), lp_y], axis=1)
    lp_y = py_print_iteration_info("lp(y)", lp_y, n, debug=debug)

    cut_off = max_target
    alpha_y = tf.cond(tf.greater(n, max_target),
                      lambda: alpha_y[:, :cut_off],
                      lambda: alpha_y)
    lp_y = tf.cond(tf.greater(n, max_target),
                   lambda: lp_y[:, :cut_off],
                   lambda: lp_y)
    lp_blank = tf.cond(tf.greater(n, max_target),
                       lambda: lp_blank[:, :cut_off],
                       lambda: lp_blank)
    alpha_blank = tf.cond(tf.greater(n, max_target),
                          lambda: alpha_blank[:, :cut_off],
                          lambda: alpha_blank)

    # all should have shape (B, n)
    blank = alpha_blank + lp_blank
    y = alpha_y + lp_y

    reduction_args = [blank, y]

    if label_rep:
      def compute_labelrep():
        col_len = tf.minimum(n - 1, max_target)
        # not first and not on diagonal
        mask = tf.logical_and(tf.greater(tf.range(col_len),  0),
                              tf.less(tf.range(col_len),  n - 2))  # (n-1,)
        mask = py_print_iteration_info("mask", mask, n, debug=debug)
        mask_exp = tf.expand_dims(mask, axis=0)  # (1, n-1)
        alpha_same = tf.where(tf.tile(mask_exp, [n_batch, 1]), prev_column, tf.ones_like(prev_column) * NEG_INF)
        alpha_same = tf.concat([alpha_same, tf.tile([[NEG_INF]], [n_batch, 1])], axis=1)
        alpha_same = tf.cond(tf.greater(n, max_target),
                             lambda: alpha_same[:, :cut_off],
                             lambda: alpha_same)

        alpha_same = py_print_iteration_info("alpha_same", alpha_same, n, debug=debug)

        labels_maxlen_same = tf.minimum(max_target - 1, n - 3)
        batchs_idxs, rows_idxs = tf.meshgrid(
          tf.range(n_batch),  # B
          tf.range(labels_maxlen_same) + 1,  # U-1
          indexing='ij'
        )
        # from (B, U, V) gather (B, N) values
        lp_same_idxs = tf.stack([batchs_idxs, rows_idxs, labels[:, :labels_maxlen_same]], axis=-1)  # (B, U-1|n-1, 3)
        lp_same_idxs = py_print_iteration_info("lp_same_idxs", lp_same_idxs, n, debug=debug)
        lp_same = tf.gather_nd(lp_column[:, :, :], lp_same_idxs)  # (B, U)
        # pad the values so we can add the scores
        # num_pads = min(2, n - labels_maxlen_same) # min(1, max_target - n + 2)
        # print("num_pads", num_pads)
        lp_same = tf.concat([tf.tile([[NEG_INF]], [n_batch, 1]),
                             lp_same,
                             tf.tile([[NEG_INF]], [n_batch, 2])], axis=1)
        lp_same = tf.cond(tf.greater(n, max_target),
                          lambda: lp_same[:, :cut_off],
                          lambda: lp_same)

        return alpha_same + lp_same

      same = tf.cond(tf.greater(n, 3),
                     compute_labelrep,
                     lambda: tf.ones_like(y) * NEG_INF)
      reduction_args += [same]

    # We can compute the most probable path
    if with_alignment:
      bt_ta = args[0]
      # reduction_args: (3|2, B, N)
      argmax_idx = tf.argmax(reduction_args, axis=0)  # (B, N) -> [2|3]
      max_len = tf.shape(reduction_args[0])[1]
      u_ranged = tf.tile(tf.expand_dims(tf.range(max_len), axis=0), [n_batch, 1])  # (1, U|n)
      u_ranged_shifted = u_ranged - 1
      # we track the state where the arc came from:
      # bt_mat: (B, T, U)           blank (u)           emit (u-1)           same (u)
      # blank_tiled = np.tile([[blank_index]], [n_batch, 1])  # (B, 1)
      # labels_exp = np.concatenate([labels, blank_tiled], axis=1)
      # u_ranged_shifted = u_ranged - 1
      # labels_emit = labels_exp[np.arange(n_batch), u_ranged_shifted]  # labels[u-1]
      # labels_same = labels_exp[np.arange(n_batch), u_ranged_shifted]  # labels[u-1]
      # we track the state where the arc came from:
      # bt_mat: (B, T, U, 2)           blank           emit           same
      # last dimension is (state-idx, label-idx)
      # sel_blank = np.stack([u_ranged, np.tile(blank_tiled, [1, max_len])], axis=-1)  # (B,)
      # sel_emit = np.stack([u_ranged_shifted, labels_emit], axis=-1)  # (1,U|n) | (B, U|n)-> (B, n, 2)
      # sel_same = np.stack([u_ranged, labels_same], axis=-1)
      # sel = np.where((argmax_idx == 0)[..., np.newaxis],
      #                sel_blank,  # blank
      #                np.where((argmax_idx == 1)[..., np.newaxis],
      #                         sel_emit,  # emit
      #                         sel_same))  # same
      blank_tiled = tf.tile([[blank_index]], [n_batch, 1])  # (B, 1)
      labels_exp = tf.concat([labels, blank_tiled], axis=1)  # (B, U+1)
      # label_idxs = tf.stack([
      #   tf.range(n_batch),
      #   tf.range(max_len)-1,
      # ], axis=-1)  # (B, 2)
      b, r = tf.meshgrid(
        tf.range(n_batch),
        tf.maximum(0, tf.range(max_len)-1),
        indexing='ij'
      )
      label_idxs = tf.stack([b, r], axis=-1)
      labels_emit = tf.gather_nd(labels_exp, label_idxs)  # (B, n)  labels[u-1]
      labels_same = labels_emit
      sel_blank = tf.stack([u_ranged, tf.tile(blank_tiled, [1, max_len])], axis=-1)
      sel_emit = tf.stack([u_ranged_shifted, labels_emit], axis=-1)  # (B, n) | (B,)
      sel_same = tf.stack([u_ranged, labels_same], axis=-1)

      argmax_idx_tiled = tf.tile(argmax_idx[..., tf.newaxis], [1, 1, 2])  # (B, n, 2)
      sel = tf.where(tf.equal(argmax_idx_tiled, 0),
                     sel_blank,  # blank
                     tf.where(tf.equal(argmax_idx_tiled, 1),
                              sel_emit,  # emit
                              sel_same))  # same
                     # u_ranged, u_ranged_shifted)  # (B, U|n)
      # we need to pad so we can stack the TA later on (instead of triangular shape)
      sel_padded = tf.pad(sel, [[0, 0], [0, max_target - max_len], [0, 0]])
      bt_ta = bt_ta.write(n-1, sel_padded)
    else:
      bt_ta = None

    red_op = tf.stack(reduction_args, axis=0)  # (2, B, N)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_column = tf.math.reduce_logsumexp(red_op, axis=0)  # (B, N)

    new_column = new_column[:, :n]
    new_column = py_print_iteration_info("new_column", new_column, n, debug=debug)
    ret_args = [n + 1, alpha_ta.write(n, new_column)]
    return ret_args + [bt_ta] if with_alignment else ret_args

  n = tf.constant(2)
  initial_vars = [n, alpha_ta]
  if with_alignment:
    initial_vars += [bt_ta]
  final_loop_vars = tf.while_loop(cond, body_forward,
                                  initial_vars,
                                  parallel_iterations=1,  # iterative computation
                                  name="rna_loss")
  if with_alignment:
    final_n, alpha_out_ta, bt_ta = final_loop_vars
    bt_mat = tf.transpose(bt_ta.stack(), [1, 0, 2, 3])  # (T, B, U, 2) -> (B, T, U, 2)
    alignments = compute_alignment_tf(bt_mat, input_lengths, label_lengths+1)
  else:
    final_n, alpha_out_ta = final_loop_vars
  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  # ll_tf = final_alpha[n_time-1, n_target-1]

  # (B,): batch index -> column index
  col_idxs = input_lengths + 1   # (B,)

  # (B,): batch index -> index within column
  within_col_idx = label_lengths
  within_col_idx = tf.where(tf.less_equal(label_lengths, input_lengths),
                            within_col_idx,  # everything ok, T>U
                            tf.ones_like(within_col_idx) * -1)  # U > T, not possible in RNA

  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=True,
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_columns",
  )
  tf_neg_inf = tf.constant(NEG_INF)

  def ta_read_body(i, res_loop_ta):
    """Reads from the alpha-columns TensorArray. We need this because of the inconsistent shapes in the TA."""
    ta_item = alpha_out_ta.read(col_idxs[i])[i]
    elem = tf.cond(tf.equal(within_col_idx[i], -1), lambda: tf_neg_inf, lambda: ta_item[within_col_idx[i]])
    elem = py_print_iteration_info("FINAL", elem, i, "col_idxs", col_idxs, "within_col_idx:", within_col_idx,
                                   "column", ta_item, debug=debug)
    return i+1, res_loop_ta.write(i, elem)

  _, ll_ta = tf.while_loop(
    lambda i, res_ta: i < n_batch,
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  if with_alignment:
    return ll_ta.stack(), alignments
  else:
    return ll_ta.stack()


def test_impl(name, acts, labels, blank_index, input_lens=None, label_lens=None,
              timing=False, debug=False, log_probs=None, label_rep=False, with_alignment=True):
  """
  runs the different implementations on the same data, comparing them.

  :param name: test name
  :param np.ndarray acts: (B, T, U, V) or (T, U, V)
  :param np.ndarray labels: (B, U) or (U,)
  :param int blank_index:
  :param label_lens: (B,)|None
  :param input_lens: (B,)|None
  :param timing:
  :param debug:
  :return: costs, grads
  """
  if acts is not None:
    if len(acts.shape) == 3:  # single -> batched
      assert len(labels.shape) == 1
      acts = np.expand_dims(acts, axis=0)  # (B, T, U, V)
      labels = np.expand_dims(labels, axis=0)  # (B, U)
      n_batch, n_time, max_target, n_vocab = acts.shape

      assert input_lens is None
      assert label_lens is None
      input_lens = np.array([n_time])  # (B,)=(1,)
      label_lens = np.array([max_target-1])
    assert len(acts.shape) == 4  # (B, T, U+1, V)
  assert input_lens is not None
  assert label_lens is not None
  if acts is None:
    assert log_probs is not None
    assert len(log_probs.shape) == 4
    n_batch, n_time, n_target, n_vocab = log_probs.shape
  else:
    assert log_probs is None
    n_batch, n_time, n_target, n_vocab = acts.shape
    log_probs = log_softmax(acts, axis=-1)  # along vocabulary

  labels = labels.astype(np.int32)
  input_lens = input_lens.astype(np.int32)
  label_lens = label_lens.astype(np.int32)

  def print_results(name, cost, grads):
    """Prints the results of an implementation."""
    cost = np.sum(cost)
    if grads is None:
      grads_msg = "n/a"
    else:
      if isinstance(grads, np.ndarray):
        if len(grads.shape) == 4:
          *grads, = grads  # list[np.ndarray]
      grads_msg = "%.4f" % sum([np.linalg.norm(grad) for grad in grads])
    print(colored("%20s" % name, "red"),
          "implementation: log-posterior=%.4f, |grads|=%s" % (
            cost, grads_msg))
  print("Test", colored("%s" % name, "yellow"), "(labelrep=%s)" % label_rep)

  list_ll = []
  list_grads = []
  list_alphas = []
  list_cost_naive = []
  list_alignments = []
  for i in range(n_batch):
    log_probs_i = log_probs[i, :input_lens[i], :label_lens[i]+1, :]
    labels_i = labels[i, :label_lens[i]]
    input_len = input_lens[i]
    assert log_probs_i.shape == (input_len, labels_i.shape[0] + 1, n_vocab)
    alphas, ll_forward = forward_pass(log_probs_i, labels_i, blank_index)
    betas, ll_backward = backward_pass(log_probs_i, labels_i, blank_index)
    assert np.allclose(ll_forward, ll_backward, atol=1e-12, rtol=1e-12), "Log-likelihood from forward and backward " \
                                                                         "pass mismatch. "
    analytical_grads = analytical_gradient(log_probs_i, alphas, betas, labels_i, blank_index)
    # enable for smaller tests, too expensive for bigger ones
    # numerical_grads = numerical_gradient(log_probs_i, labels_i, -ll_forward, blank_index)
    # assert np.allclose(analytical_grads, numerical_grads, atol=1e-6, rtol=1e-6), "Analytical and numerical " \
    #                                                                              "computation of gradient mismatch. "

    list_ll.append(-ll_forward)
    if debug:
      print("i=%2d:" % i, "T=%d, U=%d" % (input_lens[i], label_lens[i]),
            "NLL", -ll_forward, "from: probs=", log_probs_i.shape,
            "and labels=", labels_i.shape)
    list_alphas.append(alphas)
    list_grads.append(analytical_grads)

    res = numpy_forward_naive(log_probs_i, labels_i, blank_index,
                              label_rep=label_rep, with_alignment=with_alignment,
                              debug=debug)
    if with_alignment:
      alpha_np_naive, cost_np_naive, alignment_naive = res
      if debug or True:
        print("[np naive] alignments:", alignment_naive)
      list_alignments += [alignment_naive]
    else:
      alpha_np_naive, cost_np_naive = res
    if not label_rep:
      np.testing.assert_almost_equal(cost_np_naive, -ll_forward, decimal=5, err_msg="costs(numpy) != costs(ref)")
    list_cost_naive += [cost_np_naive]
  if debug:
    print("analytical == numerical grad: %s" % colored("MATCH", "green"))
  costs_ref = np.stack(list_ll, axis=0)
  if label_rep:
    print_results("NumPy naive", list_cost_naive, None)
  else:
    print_results("Reference", costs_ref, list_grads)

  res_np = numpy_forward_batched(log_probs, labels, blank_index=blank_index,
                                 input_lens=input_lens, label_lens=label_lens,
                                 label_rep=label_rep, with_alignment=with_alignment,
                                 debug=debug)
  if with_alignment:
    costs_np, alignments_np = res_np
    if debug or True:
      print("[np batched] alignments:", alignments_np)
    for i in range(n_batch):
      np.testing.assert_allclose(alignments_np[i][:input_lens[i]], list_alignments[i])
  else:
    costs_np = res_np

  print_results("NumPy batched", costs_np, None)
  if label_rep:
    np.testing.assert_almost_equal(costs_np, list_cost_naive, decimal=5,
                                   err_msg="costs(numpy batched) != costs(numpy naive)")
  else:
    np.testing.assert_almost_equal(costs_np, costs_ref, decimal=5, err_msg="costs(numpy) != costs(ref)")

  with sess.as_default():
    labels_ph = tf.compat.v1.placeholder(tf.int32, [None, None])
    log_probs_ph = tf.compat.v1.placeholder(tf.float32, [None, None, None, None])
    input_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])
    label_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])
    tf_run_opts = {}
    if timing:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      tf_run_opts = {'options': run_options, 'run_metadata': run_metadata}
    with tf.device("/cpu:0"):  # run on CPU, so that TF doesn't hog the GPU
      ret_ph = tf_forward_shifted_rna(log_probs_ph, labels_ph,
                                      input_lengths=input_lengths_ph,
                                      label_lengths=label_lengths_ph,
                                      blank_index=blank_index,
                                      label_rep=label_rep, with_alignment=with_alignment,
                                      debug=debug)
      if with_alignment:
        costs_ph, alignments_ph = ret_ph
      else:
        costs_ph = ret_ph
      grads_ph = tf.gradients(xs=log_probs_ph, ys=[-costs_ph])
    assert len(grads_ph) == 1
    grads_ph = grads_ph[0]
    ll_tf, grads_tf, alignments_tf = sess.run([costs_ph, grads_ph] + ([alignments_ph] if with_alignment else [None]),
                                              feed_dict={log_probs_ph: log_probs,
                                                         labels_ph: labels,
                                                         input_lengths_ph: input_lens,
                                                         label_lengths_ph: label_lens}, **tf_run_opts)
    if timing:
      max_bytes = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
      print("max bytes in use:", max_bytes)
      from tensorflow.python.client import timeline
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline_tf_impl_%s.json' % name, 'w') as f:
        f.write(ctf)
    nll_tf = -ll_tf
  print_results("Tensorflow", nll_tf, grads_tf)

  if with_alignment:
    if debug or True:
      print("[TF] alignments:", alignments_np)
    np.testing.assert_equal(alignments_tf, alignments_np)

  assert np.isfinite(grads_tf).all(), "Found non-finite values in TF gradients."
  # Do all the tests (ref vs TF), for score, and grads
  if label_rep:
    np.testing.assert_almost_equal(nll_tf, list_cost_naive, decimal=3, err_msg="costs(TF) != costs(numpy naive)")
    if debug:
      print("TF vs Naive Numpy: log posterior ", colored("MATCH", "green"))
  else:
    np.testing.assert_almost_equal(nll_tf, costs_ref, decimal=3, err_msg="costs(TF) != costs(ref)")
    if debug:
      print("TF vs Reference: log posterior ", colored("MATCH", "green"))
    for i in range(n_batch):
      np.testing.assert_almost_equal(
          grads_tf[i, :input_lens[i], :label_lens[i]+1],
          list_grads[i], decimal=4)
    if debug:
      print("TF vs Reference: gradients     ", colored("MATCH", "green"))

  if not label_rep:
    try:
      from warp_rna import rna_loss
      import torch
      log_probs_pt = torch.from_numpy(log_probs).float()
      log_probs_pt.requires_grad_(True)
      costs_pt = rna_loss(log_probs_pt.cuda(),
                          torch.from_numpy(labels).cuda(),
                          torch.from_numpy(input_lens).cuda(),
                          torch.from_numpy(label_lens).cuda(),
                          average_frames=False, blank=blank_index)
      costs_pt.sum().backward()
      grads_pt = log_probs_pt.grad
      print_results("PyTorch", costs_pt.detach().cpu().numpy(), grads_pt.numpy())
      np.testing.assert_almost_equal(nll_tf, costs_pt.detach().cpu().numpy(), decimal=3,
                                     err_msg="costs(TF) != costs(PyTorch)")
      np.testing.assert_almost_equal(grads_tf, grads_pt.numpy(), decimal=3,
                                     err_msg="grads(TF) != grads(PyTorch)")
    except ImportError:
      print(colored("%20s" % "PyTorch", "red"),
            "implementation: %s" % colored("module not found.", "yellow"))
      print("")
  print()
  return nll_tf, grads_tf


def test_small():
  """Small test, copied from
    https://github.com/awni/transducer/blob/master/ref_transduce.py
  """
  blank_index = 200
  vocab_size = 201
  input_len = 5
  output_len = 4
  acts = np.random.rand(input_len, output_len, vocab_size)
  labels = np.array([23, 44, 92])
  test_impl("test_small", acts, labels, blank_index, with_alignment=True)


def test_small_labelrep():
  """Small test, label repetititions enabled.
  """
  blank_index = 0
  vocab_size = 4
  input_len = 5
  output_len = 4
  acts = np.random.rand(input_len, output_len, vocab_size)
  labels = np.array([1, 2, 3])
  test_impl("test_small", acts, labels, blank_index, label_rep=True)


def test_size_t_greater_u():
  """Tests for case when T > 2*U"""
  blank_index = 0
  n_time = 20
  n_target = 3
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  test_impl("test_size: T>U", acts, labels, blank_index)


def test_size_t_equal_u():
  """Tests for case when T == U"""
  blank_index = 0
  n_time = 7
  n_target = 7
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.random.randint(1, n_vocab-1, (n_target-1,))
  test_impl("test_size: U==T", acts, labels, blank_index)


def test_sizes():
  """Tests for case when T == U"""
  blank_index = 0
  n_vocab = 5

  for n_time in range(3, 12):
    for n_target in range(3, n_time+2):
      acts = np.random.random_sample((n_time, n_target, n_vocab))
      labels = np.random.randint(1, n_vocab-1, (n_target-1,))
      test_impl("test_sizes: T=%d, U=%d" % (n_time, n_target), acts, labels, blank_index)


def test_blank_idx_nonzero():
  """Tests when the blank-idx is not 0."""
  n_time = 8
  n_target = 3
  n_vocab = 5
  acts = np.random.standard_normal((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  for blank_index in range(n_vocab):
    test_impl("test_blank_idx (%d)" % blank_index, acts, labels, blank_index=blank_index)


def test_real():
  blank_index = 1030
  fpath = "/work/data/debug-rna-impl/debug-globalstep529839.npz"
  if not os.path.exists(fpath):
    print("Skipping test 'real' due to missing file '%s'." % fpath)
    return
  item = np.load(fpath)
  log_probs = item["log_probs"]  # (B, T, U+1, V)

  n_batch, n_time, n_target, n_vocab = log_probs.shape
  log_probs = np.concatenate([log_probs, np.random.random((n_batch, n_time, 1, n_vocab))], axis=2)  # add +1 to outputlen, bug in config!!!
  assert log_probs.shape == (n_batch, n_time, n_target+1, n_vocab)

  targets = item["targets"]

  enc_lens = item["enc_lens"]
  dec_lens = item["dec_lens"]

  # print("enc lens  :", enc_lens)
  # print("dec lens  :", dec_lens)
  # print("targets   :", targets.shape)
  # print("log probs :", log_probs.shape)

  assert np.isfinite(log_probs).all()

  assert enc_lens.shape == (n_batch,)
  assert dec_lens.shape == (n_batch,)
  assert targets.shape == (n_batch, n_target)
  test_impl("test_real", None, labels=targets, blank_index=blank_index, input_lens=enc_lens,
            label_lens=dec_lens, timing=False, log_probs=log_probs)


def test_batched():
  """Check batched, different output/input lengths.
  """
  n_batch = 8
  n_time = 15
  n_target = 7
  n_vocab = 5
  acts = np.random.standard_normal((n_batch, n_time, n_target, n_vocab))
  for i in [3]: #range(8):
    label_lengths = np.random.randint(1, n_target, (n_batch,))  # [1, U)
    input_lengths = label_lengths + np.random.randint(1, n_time - n_target, (n_batch,))
    labels = np.random.randint(1, n_vocab-1, (n_batch, n_target-1,))  # except blank=0
    test_impl("batched(%d): T=%r, U=%r" % (i, input_lengths, label_lengths), acts, labels, blank_index=0, input_lens=input_lengths,
              label_lens=label_lengths, timing=False, with_alignment=True, debug=False)


def test_batched_labelrep():
  """Check batched, different output/input lengths.
  """
  n_batch = 8
  n_time = 15
  n_target = 7
  n_vocab = 5
  acts = np.random.standard_normal((n_batch, n_time, n_target, n_vocab))
  for i in range(8):
    label_lengths = np.random.randint(1, n_target, (n_batch,))  # [1, U)
    input_lengths = label_lengths + np.random.randint(0, n_time-n_target, (n_batch,))
    labels = np.random.randint(1, n_vocab-1, (n_batch, n_target-1,))  # except blank=0
    test_impl("batched(%d): T=%r, U=%r" % (i, input_lengths, label_lengths), acts, labels, blank_index=0, input_lens=input_lengths,
              label_lens=label_lengths, label_rep=True)


def test_batched_tiled():
  """
  Tiled across batch-dim, so that every i in the batch is the same.
  Except for the label/input lengths.
  Better for debugging.
  """
  n_batch = 8
  n_time = 4
  n_target = 3
  n_vocab = 5
  acts = np.random.standard_normal((1, n_time, n_target, n_vocab))
  acts = np.tile(acts, [n_batch, 1, 1, 1])
  for i in range(8):
    label_lengths = np.random.randint(1, n_target, (n_batch,))  # [1, U)
    input_lengths = np.array([n_time] * (n_batch-1))
    input_lengths = np.concatenate([input_lengths, [n_time-1]])
    # labels = np.random.randint(1, n_vocab-1, (n_batch, n_target-1,))  # except blank=0
    labels = np.random.randint(1, n_vocab - 1, (1, n_target - 1,))  # except blank=0
    labels = np.tile(labels, [n_batch, 1])
    test_impl("batched(%d): T=%r, U=%r" % (i, input_lengths, label_lengths), acts, labels, blank_index=0, input_lens=input_lengths,
              label_lens=label_lengths, timing=False)


def test_big():
  """Big test, with timing.
  """
  blank_index = 0
  vocab_size = 50
  input_len = 200
  output_len = 40
  acts = np.random.rand(input_len, output_len, vocab_size)
  labels = np.random.randint(1, vocab_size, output_len-1)
  test_impl("test_big", acts, labels, blank_index, timing=True)


def run_all_tests():
  test_batched()
  test_batched_labelrep()
  test_real()
  test_batched_tiled()
  test_small()
  # test_small_labelrep()
  # test_size_t_greater_u()
  # test_size_t_equal_u()
  # test_sizes()
  test_blank_idx_nonzero()
  # test_big()


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  sess = tf.Session()
  import better_exchook
  better_exchook.install()

  np.random.seed(42)

  run_all_tests()