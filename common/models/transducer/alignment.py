from returnn.tf.util.basic import get_shape_dim, check_input_dim
from returnn.tf.util.data import Data
from returnn.tf.layers.basic import LayerBase
import tensorflow as tf
from typing import List


def rna_alignment(source, **kwargs) -> tf.Tensor:
  """ Used only to create alignments according to RNA loss function.
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      source: function (i: int, as_data: bool = False, ...) -> tf.Tensor|Data
              which returns one of:
        output_log_prob: [B, T, U+1, V] log-probabilities
        real_target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      alignment: [B, T] which holds a value from interval (0:blank_ix) for each alignment frame
  """
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
  _, alignment = tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                        blank_index=blank_idx, debug=False, with_alignment=True)
  return alignment  # [B, T]


def rnnt_alignment(source, **kwargs) -> tf.Tensor:
  """ Used only to create alignments according to RNNT loss function.
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      source: function (i: int, as_data: bool = False, ...) -> tf.Tensor|Data
              which returns one of:
        output_log_prob: [B, T, U+1, V] log-probabilities
        real_target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      alignment: [B, T+U] which holds a value from interval (0:blank_ix) for each alignment frame
  """
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
  return alignment  # [B, T+U]


def rna_alignment_out_type(sources: List[LayerBase], **_kwargs) -> Data:
  """ Computes the rna-alignment Data_out_type for RNA alignment
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      sources:
        output_log_prob: [B, T, U+1, V] log-probabilities
        real_target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      alignment [B, T]
  """
  return Data(name="rna_alignment_output", sparse=True, dim=sources[0].output.dim,
              size_placeholder={0: sources[2].output.size_placeholder[0]})


def rnnt_alignment_out_type(sources: List[LayerBase], **_kwargs) -> Data:
  """ Computes the rnnt-alignment Data_out_type for RNNT alignment
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      sources:
        output_log_prob: [B, T, U+1, V] log-probabilities
        real_target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      alignment [B, T+U]
  """
  return Data(name="rnnt_alignment_output", sparse=True, dim=sources[0].output.dim,
              size_placeholder={0: sources[1].output.size_placeholder[0] + sources[2].output.size_placeholder[0]})
