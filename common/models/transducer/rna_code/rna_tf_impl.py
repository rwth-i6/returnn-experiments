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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "returnn"))

NEG_INF = -float("inf")


def py_print_iteration_info(msg, var, n, *vars, debug=True):
  """adds a tf.print op to the graph while ensuring it will run (when the output is used)."""
  if not debug:
    return var
  var_print = tf.print("n=", n, "\t", msg, tf.shape(var), var, *vars,
                       summarize=-1, output_stream=sys.stdout)
  with tf.control_dependencies([var_print]):
    var = tf.identity(var)
  return var


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
    # this is ugly, but works on both TF 1.15 (does not support broadcasting) and 2.3
    idx = tf.where(tf.tile(tf.greater(t, input_lens - 1)[:, None], [1, 2]), initial_idx, idx)
    return t-1, alignments, label_align, idx
  t = max_time-2
  final_t, final_alignments, final_label_align_ta, final_idx = tf.while_loop(lambda t, idx, _, _2: tf.greater_equal(t, 0),
                                                       body,
                                                       (t, alignments, label_align, initial_idx))

  final_label_alignment = final_label_align_ta.stack()
  return tf.transpose(final_label_alignment)


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
    element_shape=(None, None),  # (B, U|n)
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

    # prev_column = alpha_ta.read(n-1)[:, :n-1]  # (B, n-1)
    column_maxlen = tf.reduce_min([max_target, n])
    prev_column = alpha_ta.read(n - 1)[:, :column_maxlen]  # (B, n-1)
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
    lambda i, res_ta: tf.less(i, n_batch),
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  if with_alignment:
    return ll_ta.stack(), alignments
  else:
    return ll_ta.stack()


def rna_loss_gather(log_probs, labels, input_lengths=None, label_lengths=None, blank_index=0,
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
  assert not label_rep, "Not implemented"
  shape = tf.shape(log_probs)
  n_batch = shape[0]     # B
  max_time = shape[1]    # T
  max_target = shape[2]  # U+1
  num_columns = max_time + 2

  labels = py_print_iteration_info("labels", labels, 0, debug=debug)
  batchs, rows, cols = tf.meshgrid(
    tf.range(n_batch),
    tf.range(max_time),
    tf.range(max_target),
    indexing='ij'
  )
  targets_exp_filed = tf.tile(labels[:, tf.newaxis, :], [1, max_time, 1])  # (B, T, U)
  # such that the dimensions align, the last target (0) will never be accessed.
  targets_exp_filed = tf.concat([targets_exp_filed, tf.zeros((n_batch, max_time, 1), dtype=tf.int32)],
                                axis=-1)  # (B, T, U+1)
  lp_y_idxs = tf.stack([batchs,
                        rows,
                        cols,
                        targets_exp_filed
                        ], axis=-1)  # (B, T, U+1, 4)
  lp_blank_idxs = tf.stack([
    batchs, rows, cols,
    tf.ones((n_batch, max_time, max_target), dtype=tf.int32) * blank_index
  ], axis=-1)

  # TODO: combine both gather_nd calls.

  lp_y = tf.gather_nd(log_probs, lp_y_idxs)  # (B, T, U)
  lp_blank = tf.gather_nd(log_probs, lp_blank_idxs)  # (B, T, U)

  lp_gather = tf.stack([lp_y, lp_blank], axis=-1)  # (B, T, U, 2)
  # better time-first for time-sync algorithm
  lp_gather_t = tf.transpose(lp_gather, [1, 0, 2, 3])  # (T, B, U, 2)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_columns,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, 2),  # (B, U, 2)
    name="log_probs",
  )
  # (B, T, U, V) -> [(B, U, V)] * (T)
  log_probs_ta = log_probs_ta.unstack(lp_gather_t)

  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_columns,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None),  # (B, U)
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

    lp_column = log_probs_ta.read(n-2)[:, :n-1, :]  # (B, U|n, 2), y|blank
    lp_column = py_print_iteration_info("lp_column", lp_column, n, debug=debug)

    column_maxlen = tf.reduce_min([max_target, n])
    prev_column = alpha_ta.read(n - 1)[:, :column_maxlen]  # (B, n-1)
    prev_column = py_print_iteration_info("prev_column", prev_column, n, debug=debug)

    alpha_blank = prev_column  # (B, N)
    alpha_blank = tf.concat([alpha_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)

    # (B, U, V) -> (B, U)
    lp_blank = lp_column[:, :, 1]  # (B, U)
    lp_blank = tf.concat([lp_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    lp_blank = py_print_iteration_info("lp(blank)", lp_blank, n, debug=debug)

    # (B,N-1) ; (B,1) ->  (B, N)
    alpha_y = prev_column
    alpha_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), alpha_y], axis=1)
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    # labels_maxlen = tf.minimum(max_target-1, n-1)
    # labels_shifted = labels[:, :labels_maxlen]  # (B, U-1|n-1)
    # labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    # batchs, rows = tf.meshgrid(
    #   tf.range(n_batch),
    #   tf.range(labels_maxlen),
    #   indexing='ij'
    # )
    # lp_y_idxs = tf.stack([batchs, rows, labels_shifted], axis=-1)  # (B, U-1|n-1, 3)
    # lp_y_idxs = py_print_iteration_info("lp_y_idxs", lp_y_idxs, n, debug=debug)
    # lp_y = tf.gather_nd(lp_column[:, :, :], lp_y_idxs)  # (B, U)
    # (B, U) ; (B, 1) -> (B, U+1)
    lp_y = lp_column[:, :, 0]  # (B, U)
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
      blank_tiled = tf.tile([[blank_index]], [n_batch, 1])  # (B, 1)
      labels_exp = tf.concat([labels, blank_tiled], axis=1)  # (B, U+1)
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
    lambda i, res_ta: tf.less(i, n_batch),
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  if with_alignment:
    return ll_ta.stack(), alignments
  else:
    return ll_ta.stack()
