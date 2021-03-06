
from returnn.tf.util.data import Data


def rnnt_loss(source, **_kwargs):
  """
  Computes the RNN-T loss function.

  :param log_prob:
  :return:
  """
  # acts: (B, T, U + 1, V)
  # targets: (B, T)
  # input_lengths (B,)
  # label_lengths (B,)
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


def rnnt_loss_out_type(**_kwargs) -> Data:
  return Data(name="rnnt_loss", shape=())
