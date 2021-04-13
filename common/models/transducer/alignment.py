from returnn.tf.util.basic import get_shape_dim, check_input_dim


def rna_alignment(source, **kwargs):
  """
  Used only to create alignments according to RNA loss function.
  :sources: [output_log_prob, real_target, "base:encoder"]
  :return: alignments: [B, T] for each frame a value in [0:blank_ix]
  """
  # acts: (B, T, U, V)
  # targets: (B, U-1)
  # input_lengths (B,)
  # label_lengths (B,)
  from .rna_align_sum_max_pure_tf import tf_forward_shifted_rna

  log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  target_len = get_shape_dim(targets.get_placeholder_as_batch_major(), 1)
  log_probs = check_input_dim(log_probs, 2, target_len + 1)

  # enc_lens = tf.Print(enc_lens, ["enc_lens:", enc_lens, "dec_lens:", dec_lens,
  #                     "targets:", tf.shape(targets.get_placeholder_as_batch_major()),
  #                     "log-probs:", tf.shape(log_probs.get_placeholder_as_batch_major())], summarize=-1)

  blank_idx = targets.dim  # targets is without blank
  costs, alignment = tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                            blank_index=blank_idx, debug=False, with_alignment=True)
  return alignment  # (B, T)


def rnnt_alignment(source, **kwargs):
  """
  Used only to create alignments according to RNNT loss function.
  :sources: [output_log_prob, real_target, "base:encoder"]
  :return: alignments: [B, T] for each frame a value in [0:blank_ix]
  """
  # alignment-length (B,T+U+1)
  # acts: (B, T, U+1, V)
  # targets: (B, U)
  # input_lengths (B,)
  # label_lengths (B,)
  from .rnnt_align_sum_max_pure_tf import tf_forward_shifted_rnnt

  log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  target_len = get_shape_dim(targets.get_placeholder_as_batch_major(), 1)
  log_probs = check_input_dim(log_probs, 2, target_len + 1)

  # alignment_length = source(0, as_data=True, auto_convert=False)
  # print_op = tf.print({"max(U+T)": tf.reduce_max(enc_lens+dec_lens), "alignment-length": alignment_length.get_sequence_lengths()}, summarize=-1)
  # with tf.control_dependencies([print_op]):
  # log_probs = tf.identity(log_probs)
  blank_idx = targets.dim
  _, alignment = tf_forward_shifted_rnnt(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                         blank_index=blank_idx, debug=False, with_alignment=True)
  return alignment  # (B, T)
