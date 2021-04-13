from .loss import (rnnt_loss, rnnt_loss_out_type,
                   rnnt_tf_loss, rna_tf_loss, rna_loss_out_type)
from .alignment import (rnnt_alignment, rnnt_alignment_out_type,
                        rna_alignment, rna_alignment_out_type)


class Topology:
  """
  Hold informations about different label topologies such as loss-, alignment-funcion and their out_types.
  loss and alignment functions are to be used in eval like layers that return a source function.

  The loss, alignment_out_type and alignment function all are to be used in EvalLayers.
  taking from layers that output the followings:
        output_log_prob: [B, T, U+1, V] log-probabilities
        target: [B, U] -> [V] target labels
        base:encoder: [B, T, Feat] -> [V] encoder output
  where
        B: batch, T: time, U:target/labels, V: vocabulary, U': seq_len of alignment
  EvalLayer offers a source() callback, which has to be used to get the mentioned data.
  """
  def __init__(self,
               name: str,
               loss,
               loss_out_type,
               alignment,
               alignment_out_type):
    """ Label Topology such as rnnt, rna, ctc.
    Args:
        loss: function (source: (i: int, as_data: bool = False, ...) -> tf.Tensor|Data, ...) -> tf.Tensor[B]
        loss_out_type: function (...) -> Data[B]
        alignment: function (source: (i: int, as_data: bool = False, ...) -> tf.Tensor|Data, ...) -> tf.Tensor[B,U']
        alignment_out_type: function (sources: list[LayerBase], ...) -> Data[B,U']
    """
    self.name = name
    self.loss = loss
    self.loss_out_type = loss_out_type
    self.alignment = alignment
    self.alignment_out_type = alignment_out_type


rna_topology = Topology(
  name="rna",
  loss=rna_tf_loss,
  loss_out_type=rna_loss_out_type,
  alignment=rna_alignment,
  alignment_out_type=rna_alignment_out_type)

rnnt_topology = Topology(
  name="rnnt",
  loss=rnnt_tf_loss,
  loss_out_type=rnnt_loss_out_type,
  alignment=rnnt_alignment,
  alignment_out_type=rnnt_alignment_out_type,)
