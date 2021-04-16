"""
Usage:
After each stage alignments are automatically generated.

# dummy
st1 = Stage(
  make_net=Pretrain(make_net, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=5).get_network,
  stage_epochs=5,
  fixed_path=False,
  alignment_topology=rnnt_topology)
st2 = Stage(
  make_net=Pretrain(make_net, {"enc_lstm_dim": (512, 1024), "enc_num_layers": (3, 6)}, num_epochs=15).get_network,
  stage_epochs=15,
  fixed_path=True,
  alignment_topology=rnnt_topology)

# Multi stage training with pretraining
get_network = MultiStager([st1, st2]).get_network

  TODO:
    - [ ] How to save the information about the differences between alignments of different topologies.
    - [ ] Make sure that the alignments correspond to the dataset used(sequence_ordering, ..)
    - [ ] Reset option
    - [ ] How to define loops? Instead of creating Stages manually (could do a for loop)
"""

from __future__ import annotations
from ast import Str
from typing import Dict, Any, List, AnyStr

from returnn.tf.util.data import Data, DimensionTag
from returnn.config import get_global_config
from ...training.switchout import switchout_target
from ...datasets.interface import TargetConfig
from .topology import Topology, rna_topology, rnnt_topology

import tensorflow as tf
import sys
import os



class Context:
  def __init__(self, task: str, target: TargetConfig, model: str,
               name: str, alignment_topology: Topology = rnnt_topology):
    self.task = task
    self.train = (task == "train")
    self.search = (task != "train")
    self.target = target
    self.align_dir = os.path.dirname(model)
    self.name = name
    self.num_labels_nb = target.get_num_classes()
    self.num_labels_wb = self.num_labels_nb + 1
    self.blank_idx = self.num_labels_nb
    self.alignment_topology = alignment_topology


def make_align(net: Dict[str, Any],
               epoch: int,  # maybe required
               extend: bool,
               ctx: Context,
               output: str = "output",
               output_log_prob: str = "output_log_prob_wb",
               encoder: str = "encoder",
               target: TargetConfig = None):
  """
  Here we assume that the decoder is a recurent network(with unit) called `output`.
  In the "unit" `output_log_prob` should define the the log distribution over the whole vocab inkl blank.
  Otherwise "base:{encoder}" which represent the output of the encoder should be provided.
  This function extends the "unit" of the decoder with logic to create and dump fullsum alginment in .hdf files.
  Requires:
    output/unit
      output_log_prob: log distribution over the whole vocab inkl blank
      f"base:data:{target}": targets of the sequence
      base:encoder: output of the encoder
    rna_fullsum_alignment: function that performs the alignment and returns for e.g [BxT] for rna alignm.
    extend: if True the already existing .hdf alignments are extended
    ctx: holds information such as the label topology, target and path to be used for .hdf files

  Durign this step 1 subepoch passes.
  """
  align_dir = ctx.align_dir
  name = ctx.name
  if not target:
    target = TargetConfig.global_from_config()
  subnet = net[output]["unit"]
  subnet[output_log_prob]["is_output_layer"] = True
  subnet["fullsum_alignment"] = {
    "class": "eval",
    "from": [output_log_prob, f"base:data:{ctx.target.key}", f"base:{encoder}"],
    "eval": ctx.alignment_topology.alignment,
    "out_type": ctx.alignment_topology.alignment_out_type,
    "is_output_layer": True
  }

  subnet["_align_dump"] = {
    "class": "hdf_dump",
    "from": "fullsum_alignment",
    "is_output_layer": True,
    "dump_per_run": True,
    "extend_existing_file": extend,  # TODO: extend only the first time
    # dataset_name: comes from **opts of the lambda in filename
    "filename":
      (lambda **opts: "{align_dir}/align.{name}_{dataset_name}.hdf".format(align_dir=align_dir,
                                                                           name=name, **opts)),
  }
  net["#trainable"] = False  # disable training
  net["#finish_all_data"] = True
  return net


def make_fixed_path(net, ctx: Context, reset=False, switchout=True,
                    output: str = "output",
                    inited_output: str = "output_",
                    ) -> Dict:
  target = ctx.target
  blank_idx = ctx.blank_idx
  train = ctx.train
  align_dir = ctx.align_dir
  name = ctx.name
  subnet = net[output]["unit"]

  # Viterbi training allows switchout
  if train and switchout:
    net["output"]["size_target"] = target
    subnet[inited_output] = {  # SwitchOut in training
      "class": "eval", "from": "output", "eval": switchout_target,
      "eval_local": {"_targetb_blank_idx": blank_idx, "_target_num_labels": target.get_num_classes()},
      "initial_output": 0
    }
  del net["lm_input"]
  # Framewise CE loss
  subnet["output_prob"] = {
    "class": "activation", "from": "output_log_prob", "activation": "exp",
    "target": target, "loss": "ce", "loss_opts": {"focal_loss_factor": 2.0}
  }
  net.update({
    "existing_alignment": {
      "class": "reinterpret_data", "from": "data:alignment",
      "set_sparse": True,  # not sure what the HDF gives us
      "set_sparse_dim": target.get_num_classes(),
      "size_base": "encoder",  # for RNA...
    },
    # The layer name must be smaller than "t_target" such that this is created first.
    "1_targetb_base": {
      "class": "copy",
      "from": "existing_alignment",
      "register_as_extern_data": "targetb" if train else None},
  })
  # Global changes

  # Reset
  if reset:
    net["#copy_param_mode"] = "reset"

  # Chunking
  _time_red = 6
  _chunk_size = 60
  net["#config"].update({
    # ..., TODO more? e.g. maximize GPU mem util
    "chunking":  # can use chunking with frame-wise training
    (
      {"data": _chunk_size * _time_red,
       "alignment": _chunk_size},
      {"data": _chunk_size * _time_red // 2,
       "alignment": _chunk_size // 2}
    )
  })

  # Meta dataset which combines:
  #   align: FixedPath HdfDataset
  #   default: the default Dataset
  for data in ["train", "dev"]:
    net["#config"][data] = get_fixed_path_meta_dataset("train", f"{align_dir}/align.{name}_{data}.hdf", ctx)
  net["#config"]["eval_datasets"] = {
    key: get_fixed_path_meta_dataset(key, "%s/align.%s.hdf" % (align_dir, key), ctx) for key in net["#config"]["eval_datasets"]}
  _output_len_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="output-len")  # it's downsampled time
  net["#config"]["extern_data"]["alignment"] = {"dim": target.get_num_classes(),
                                                "sparse": True,
                                                "same_dim_tags_as": {"t": _output_len_tag}}
  return net


def get_fixed_path_meta_dataset(task: str,
                                path_2_hdf: str,
                                ctx: Context):
  """
  TODO:
  """
  train = ctx.train

  # TODO should be put in a metadataset together with the normal dataset
  align_dataset = {
    "class": "HDFDataset", "files": [path_2_hdf],
    "use_cache_manager": True,
    # "unique_seq_tags": True  # dev set can exist multiple times
    # TODO: otherwise not right selection
    # "seq_list_filter_file": files["segments"],
    # "partition_epoch": epoch_split,
    # TODO: do we really need the num_seq
    # "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
  }
  if train:
    # TODO: do we really need the num_seq
    # align_dataset["seq_ordering"] = "laplace:%i" % (estimated_num_seqs[data] // 1000)
    align_dataset["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"

  default_dataset = {"class": "my_default_dataset"}

  d = {
    "class": "MetaDataset",
    "datasets": {"default": default_dataset, "align": align_dataset},
    "data_map": {
      "data": ("default", "data"),
      "alignment": ("align", "data"),
    },
    "seq_order_control_dataset": "align",  # it must support get_all_tags
  }
  return d
  pass


class Stage:
  def __init__(self, make_net,
               stage_epochs: int,
               fixed_path: bool,
               alignment_topology: Topology,
               name: str = None):
    self.make_net = make_net
    self.stage_epochs = stage_epochs
    self.fixed_path = fixed_path  # False if full_sum and True if fixed_path
    self.alignment_topology = alignment_topology  # rna, rnnt or ctc topology
    if name is None:
      name = alignment_topology.name + f"_{'fixed_path' if fixed_path else 'full_sum'}"
    self.name = name  # name used to overwrite the model name for the checkpoints


class MultiStager:
  """ Wrapper around Pretrain which enables Multi-Stage training"""
  def __init__(self, stage_list: List[Stage]):
    self.stage = stage_list[0]  # saves the stage we are on
    self.index = 0  # index of current stage
    # accumulative sum of the epochs
    # so that they represent epoch up to which the stage lasts
    for i in range(len(stage_list) - 1):
      stage_list[i + 1].stage_epochs += stage_list[i].stage_epochs + 1  # accumulative sum of the epochs
    self.stage_list = stage_list

  def update(self, epoch: int):
    """ Update model for the next stage if necessary"""
    # Update context(hmm)
    task = get_global_config().value("task", "train")
    target = TargetConfig.global_from_config()
    model = get_global_config().value("model", "net-model/network")
    self.ctx = Context(task=task, target=target, model=model, name=self.stage.name,
                       alignment_topology=self.stage.alignment_topology)
    # Update model
    if len(self.stage_list) < self.index and self.stage.stage_epochs < epoch:
      self.index += 1
      self.stage = self.stage_list[self.index]

  def get_net(self, epoch):
    return self.stage.make_net(epoch)

  def get_align_net(self, epoch):
    net = self.get_net(epoch)
    return make_align(net=net, epoch=epoch, extend=False, ctx=self.ctx)

  def get_fixed_path_net(self, epoch):
    net = self.get_net(epoch)
    return make_fixed_path(net=net, ctx=self.ctx, reset=True)

  def get_network(self, epoch: int) -> Dict[str, Any]:
    """ Gets the network from the pretrainer """
    """ Builds and updates the network according to the epoch we are in now and adds alignment layer if required """
    self.update(epoch)
    if self.stage.stage_epochs == epoch:  # alignment time CE nets should do fs alignments
      net = self.get_align_net(epoch)  # add alignment_dumping_logic
    elif self.stage.fixed_path:
      net = self.get_fixed_path_net(epoch)
    else:
      net = self.get_net(epoch)

    return net
