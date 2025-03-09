"""
Usage:
After each stage alignments are automatically generated.

# dummy
st1 = Stage(
  make_net=Pretrain(make_net, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=5).get_network,
  num_epochs=2,
  fixed_path=False,
  alignment_topology=rna_topology,
)
st2 = Stage(
  make_net=Pretrain(make_net, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=3).get_network,
  num_epochs=5,
  fixed_path=True,
  stage_num_align=0,
  alignment_topology=rna_topology,
)

# Multi stage training with pretraining
get_network = TransducerMultiStager([st1, st2]).get_network

  TODO:
    - [ ] Make sure that the alignments correspond to the dataset used(sequence_ordering, ..)
    - [ ] Chunking for RNNT
    - [ ] How to define loops? Instead of creating Stages manually (could do a for loop)
"""

from __future__ import annotations
from typing import Dict, Any, List
from returnn.config import get_global_config
from ....datasets.interface import TargetConfig
from ..topology import Topology
from ..transducer_fullsum import Context
from .alignment_dumping import update_net_for_alignment_dumping
from .fixed_path_training import update_net_for_fixed_path_training
import os


class Stage:
  def __init__(self, make_net,
               num_epochs: int,
               alignment_topology: Topology,
               fixed_path: bool = False,
               reset: bool = True,
               chunking: bool = False,
               stage_num_align: int = -1,
               name: str = None):
    """Represents one stage of the transducer training pipeline
    Args:
        make_net ([type]): callback to save the method for creating the network
        num_epochs (int): nr of epochs this stage lasts
        alignment_topology (Topology): rna, rnnt or ctc label topology
        fixed_path (bool): True if it does fixed_path training.
        reset (bool): Whether to reset the weights of the network.
        chunking (bool): Whether to do chunking.
        stage_num_align (int): Stage nr which provides the alignments in case of FixedPath training.
        name (str): Name descring the stage.
    """
    self.make_net = make_net
    self.num_epochs = num_epochs
    self.fixed_path = fixed_path
    self.alignment_topology = alignment_topology
    self.reset = reset
    self.chunking = chunking
    self.stage_num_align = stage_num_align
    if name is None:
      name = alignment_topology.name + f"_{'fixed_path' if fixed_path else 'full_sum'}"
    self.name = name

  def st(self, **kwargs) -> Stage:
    """
    Creates a copy of itself where **kwargs may provide different instace variables.
    i.e `my_stage.st(fixed_path=True, stage_num_align=1)` returns `my_stage` with changed instance
    variables, fixed_path and stage_num.
    """
    import copy
    cp = copy.deepcopy(self)
    for (k, v) in kwargs.items():
      assert hasattr(cp, k), f"Stage has no {k} attribute"
      setattr(cp, k, v)
    return cp


class TransducerFullSumAndFramewiseTrainingPipeline:
  """Wrapper around get_network() which enables Multi-Stage training"""
  def __init__(self, stage_list: List[Stage]):
    self.type = "FullSum"  # type of stage. It can be one of {"FullSum", "CE", "Align"}
    self.stage = stage_list[0]  # saves the stage we are on
    self.index = 0  # index of current stage
    self.start_epoch = 0  # holds the last epoch of the previous stage(i.e our start epoch).
    self.align_dir = os.path.dirname(get_global_config().value("model", "net-model/network"))
    self.epoch_split = get_global_config().get_of_type("train", dict).get("partition_epoch", 1)
    self.stage_list = stage_list
    self._proof_check_stages()

  def _proof_check_stages(self):
    for (i, st) in enumerate(self.stage_list):
      if st.fixed_path:
        assert st.stage_num_align >= 0, f"The stage to take the alginments from is not set in stage {i}."

  def _stage_epoch(self, epoch) -> int:
    """Returns the epoch number relative to the start of current stage"""
    return epoch - self.start_epoch

  def _stage_end_epoch(self) -> int:
    """
    Returns the total number of epochs in this stage(training + alignment dumping epochs)
    That corresponds to the last epoch of the current stage.
    """
    return self.stage.num_epochs + self.epoch_split

  def _update(self, epoch: int):
    """Update model for the next stage if necessary"""
    if self.index < len(self.stage_list):  # only if we aren't already in the last stage
      self.index += 1
      self.stage = self.stage_list[self.index]

      self.start_epoch = epoch

  def _get_net(self, epoch: int) -> Dict[str, Any]:
    return self.stage.make_net(epoch)

  def _get_net_with_align_dumping(self, epoch: int, ctx: Context) -> Dict[str, Any]:
    net = self._get_net(epoch)
    epoch0 = epoch - 1
    # False only for the first align_dump epoch, True for the rest of algin_dump epochs
    extend = epoch0 % self.epoch_split > 0
    net = update_net_for_alignment_dumping(net=net, extend=extend, ctx=ctx,
                                           stage_num=self.index, align_dir=self.align_dir,
                                           alignment_topology=self.stage.alignment_topology)
    return net

  def _get_net_with_fixed_path_training(self, epoch: int, ctx: Context) -> Dict[str, Any]:
    net = self._get_net(epoch)
    net = update_net_for_fixed_path_training(net=net, ctx=ctx, align_dir=self.align_dir,
                                             stage_num_align=self.stage.stage_num_align)

    # Global changes
    # Reset
    if self.stage.reset:
      net["#copy_param_mode"] = "reset"

    # TODO: Chunking  not working for RNNT
    if self.stage.chunking:
      _time_red = 6
      _chunk_size = 60
      net["#config"].update({
        # TODO: not configurable yet. How to provide the params to the stage?
        "chunking":  # can use chunking with frame-wise training
        (
          {"data": _chunk_size * _time_red, "alignment": _chunk_size},
          {"data": _chunk_size * _time_red // 2, "alignment": _chunk_size // 2}
        )
      })

    return net

  def get_network(self, epoch: int) -> Dict[str, Any]:
    """Gets the network from the pretrainer
    Builds and updates the network according to the epoch we are in now.
    It adds alignment if required.
    """


    task = get_global_config().value("task", "train")
    target = TargetConfig.global_from_config()
    ctx = Context(task=task, target=target, beam_size=12)

    #      ............................  Stage1   ..............................
    # start+1         train     start+N     start+N+1      align_dump     start+N+EpochSplit
    #      |-|-|-|-|-|-|-|-|-|-|-|-|-|     -     |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
    if self.stage.num_epochs < self._stage_epoch(epoch) <= self._stage_end_epoch():  # Dump alignments
      self.type = "Align"
      net = self._get_net_with_align_dumping(epoch, ctx)
    elif self.stage.fixed_path:  # Train with forced alignments and CE loss(fixed path training)
      self.type = "CE"
      net = self._get_net_with_fixed_path_training(epoch, ctx)
    else:  # Train with fullsum loss
      self.type = "FullSum"
      net = self._get_net(epoch)

    # End of stage, make the jump to the next stage
    if self._stage_end_epoch() == self._stage_epoch(epoch):
      self._update(epoch)

    return net
