from returnn.tf.util.data import Data
import tensorflow as tf


def rnnt_loss(source, **_kwargs) -> tf.Tensor:
  """ Computes the RNN-T loss function. Native TF kernel implementation.
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      source: function (i: int, as_data: bool = False, ...) -> tf.Tensor|Data
              which returns one of:
        output_log_prob: [B, T, U+1, V] log-probabilities
        target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      costs: [B]
  """
  from returnn.extern.HawkAaronWarpTransducer import rnnt_loss

  log_probs = source(0, as_data=True, auto_convert=False)
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  blank_idx = targets.dim  # targets is without blank

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  costs = rnnt_loss(
    log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
    blank_label=blank_idx)
  costs.set_shape((None,))  # (B,)
  return costs


def rnnt_tf_loss(source, **kwargs) -> tf.Tensor:
  """ Computes the RNN-T loss function. Pure TF.
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      source: function (i: int, as_data: bool = False, ...) -> tf.Tensor|Data
              which returns one of:
        output_log_prob: [B, T, U+1, V] log-probabilities
        target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      costs: [B]
  """
  from .rnnt_align_sum_max_pure_tf import tf_forward_shifted_rnnt

  log_probs = source(0, as_data=True, auto_convert=False)
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  blank_idx = targets.dim

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  costs = -tf_forward_shifted_rnnt(
    log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
    blank_index=blank_idx, debug=False)
  costs.set_shape((None,))  # (B,)
  return costs


def rna_tf_loss(source, **kwargs) -> tf.Tensor:
  """ Computes the RNA loss. Pure TF.
  B: batch, T: time, U:target/labels, V: vocabulary
  Args:
      source: function (i: int, as_data: bool = False, ...) -> tf.Tensor|Data
              which returns one of:
        output_log_prob: [B, T, U+1, V] log-probabilities
        target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  Returns:
      costs: [B]
  """
  from .rna_align_sum_max_pure_tf import tf_forward_shifted_rna
  from returnn.tf.compat import v1 as tf

  log_probs = source(0, as_data=True, auto_convert=False)
  targets = source(1, as_data=True, auto_convert=False)
  encoder = source(2, as_data=True, auto_convert=False)

  enc_lens = encoder.get_sequence_lengths()
  dec_lens = targets.get_sequence_lengths()

  blank_idx = targets.dim  # targets is without blank
  costs = -tf_forward_shifted_rna(
    log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
    blank_index=blank_idx, debug=False)
  costs = tf.where(tf.is_finite(costs), costs, tf.zeros_like(costs))
  return costs


def rnnt_loss_out_type(**_kwargs) -> Data:
  return Data(name="rnnt_loss", shape=())


def rna_loss_out_type(**_kwargs) -> Data:
  return Data(name="rnna_loss", shape=())
