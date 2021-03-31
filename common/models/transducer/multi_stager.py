from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
from returnn.tf.util.data import Data
from returnn.config import get_global_config
from .rna_code.rna import rna_fullsum_alignment
from ...datasets.interface import TargetConfig
from ...training.pretrain import Pretrain


class InUseState:
  """ This clase is used to keep the state of the multi stager."""
  def __init__(self, model_epoch_list: List[Tuple[Pretrain, int]]):
    self.model = model_epoch_list[0][0].get_network(0)  # saves the model in use
    self.up_to_epoch = model_epoch_list[0][1]
    self.index = 0
    self.model_epoch_list = model_epoch_list

  def update(self, epoch: int):
    """ use next model """
    if self.up_to_epoch < epoch:
      self.index += 1
      self.up_to_epoch += self.model_epoch_list[self.index][1] + 1
    self.model = self.model_epoch_list[self.index][0].get_network(epoch)


class MultiStager:
    """ Wrapper around Pretrain which enables Multi-Stage training"""
    def __init__(self, model_epoch_list: List[Tuple[Pretrain, int]]):
      self.model_epoch_list = model_epoch_list
      self.state = InUseState(model_epoch_list)

    def make_align(self, epoch,
                   output: Optional[str] = "output",
                   output_log_prob: Optional[str] = "output_log_prob",
                   encoder: Optional[str] = "encoder",
                   target: TargetConfig = None):
      """
      Here we assume that the decoder is a recurent network(with unit) called `output`.
      In the "unit" `output_log_prob` should define the the log distribution over the whole vocab inkl blank.
      Otherwise "base:{encoder}" which represent the output of the encoder should be provided.

      This function extends the "unit" of the decoder with logic to create and dump fullsum alginment in .hdf files.

      Requires:
        "{output}/unit"
          "output_log_prob": log distribution over the whole vocab inkl blank
          f"base:data:{target}": targets of the sequenc
          "base:encoder": output of the encoder
        rna_fullsum_alignment: function that performs the alignment and returns for e.g [BxT] for rna alignm.
        model <- via get global
        EpochSplit <- via the global
        dataset_name: comes from **opts of the lambda in filename

      Usage:
      get_network = MultiStager([
                      Pretrain( make_net_FS, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=50),
                      Pretrain( make_net_CE, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=20),
                      Pretrain( make_net_FS, {"enc_lstm_dim": (1024, 1024), "enc_num_layers": (6, 6)}, num_epochs=50),
                      Pretrain( make_net_CE, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=20),
                    ]).get_network
      TODO:
        - [ ] Make sure that the alignments correspond to the dataset used(sequence_ordering, ..)
        - [ ] Are there so many configuration differences between FS and CE network? If not we could use
              a function like self.make_ce() which adds the required differences just like self.make_align().
        - [ ] How to get targetb_num_labels, how to get epoch_split? Dataset?
        - [ ] Reset option, for example we can train with CE for a while and than switch to FS with same weights
      """
      if not target:
        target = TargetConfig.global_from_config()
      epoch0 = epoch - 1

      subnet = self.state.model[output]["unit"]
      subnet["fullsum_alignment"] = {
        "class": "eval",
        "from": [output_log_prob, f"base:data:{target}", f"base:{encoder}"],
        "eval": rna_fullsum_alignment,                # TODO: how to get targetb_num_labels
        "out_type": lambda sources, **kwargs: Data(name="rna_alignment_output", sparse=True, dim=targetb_num_labels,
                                                   size_placeholder={0: sources[2].output.size_placeholder[0]}),
        "is_output_layer": True
      }

      model = get_global_config().value("model", "net-model/network")
      subnet["_align_dump"] = {
        "class": "hdf_dump",
        "from": "fullsum_alignment",
        "is_output_layer": True,
        "dump_per_run": True,
        "extend_existing_file": epoch0 % EpochSplit > 0,  # TODO: how to get epoch_split? Dataset?
        "filename": (lambda **opts: "{align_dir}/align.{dataset_name}.hdf".format(align_dir=model, **opts)),
      }
      self.state.model["#trainable"] = False  # disable training
      self.state.model["#finish_all_data"] = True

    def update_network(self, epoch: int):
      """ Updates the network according to the epoch we are in now"""
      self.state.update(epoch)
      if self.state.up_to_epoch == epoch:  # alignment time
        self.make_align(epoch)  # add algnment_dumping_logic

    def get_network(self, epoch: int) -> Dict[str, Any]:
      """ Gets the network from the pretrainer. """
      self.update_network(epoch)
      print(self.state.model, self.state.up_to_epoch, self.state.index)
      return self.state.model
