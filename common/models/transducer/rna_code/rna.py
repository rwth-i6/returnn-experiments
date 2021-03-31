from .rna_tf_impl import tf_forward_shifted_rna

def rna_loss(source, **kwargs):
  """
  Computes the RNA loss function.

  :param log_prob:
  :return:
  """
  # acts: (B, T, U, V)
  # targets: (B, U-1)
  # input_lengths (B,)
  # label_lengths (B,)
  from returnn.tf.compat import v1 as tf

  log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  blank_idx = targets.dim  # targets is without blank
  costs = -tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                  blank_index=blank_idx, debug=False)
  costs = tf.where(tf.is_finite(costs), costs, tf.zeros_like(costs))
  return costs


def rna_fullsum_alignment(source, **kwargs):
  """
  Computes the RNA loss function. Used only to create alignments.
  :inputs output_log_prob, real_target, "base:encoder"
  :param log_prob:
  :return: alignments: [B, T] for each frame a value in [0:blank_ix]
  """
  # acts: (B, T, U, V)
  # targets: (B, U-1)
  # input_lengths (B,):
  # label_lengths (B,)

  from returnn.tf.util.basic import get_shape_dim, check_input_dim

  log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  target_len = get_shape_dim(targets.get_placeholder_as_batch_major(), 1)
  log_probs = check_input_dim(log_probs, 2, target_len + 1)

  blank_idx = targets.dim  # targets is without blank
  costs, alignment = tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                            blank_index=blank_idx, debug=False, with_alignment=True)
  return alignment  # (B, T)
