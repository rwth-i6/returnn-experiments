from __future__ import annotations
from typing import Dict, Any
from returnn.tf.util.data import DimensionTag
from returnn.config import get_global_config
from ....training.switchout import switchout_target
from .pipeline import Context


def update_net_for_fixed_path_training(net: Dict[str, Any],
                                       ctx: Context,
                                       stage_num_align: int,
                                       align_dir: str,
                                       switchout: bool = True,
                                       decoder: str = "output",
                                       inited_switchout_output: str = "output_",
                                       output_log_prob_wb: str = "output_log_prob_wb"
                                       ) -> Dict[str, Any]:
  """
  Args:
      net (Dict[str, Any]): [description]
      ctx (Context): context providing us with extra info.
      stage_num_align (int): Stage number from which we take the dumped alignments.
      align_dir (str): Path to the folder with the .hdf files.
      switchout (bool): Whether to do switchout on the predicted labels.
      decoder (str): Whether to do switchout on the predicted label.
      inited_switchout_output (str): from this output_is_not_blank is calculated.

  Returns:
      Dict[str, Any]: [updated network dictionary]
  """
  new_target_name = "targetb"
  target = ctx.target
  blank_idx = ctx.blank_idx
  train = ctx.train

  subnet = net[decoder]["unit"]
  subnet["target"] = new_target_name

  if train:
    subnet["size_target"] = new_target_name
    del subnet["lm_input"]
    del subnet["full_sum_loss"]
    if switchout:  # Framewise training allows switchout
      subnet[inited_switchout_output] = {  # SwitchOut in training
        "class": "eval", "from": "output", "eval": switchout_target,
        "eval_local": {"targetb_blank_idx": blank_idx, "target_num_labels": target.get_num_classes()},
        "initial_output": 0
      }

  # The layer name must be smaller than "t_target" such that this is created first.
  net["existing_alignment"] = {"class": "reinterpret_data",
                               "from": "data:alignment",
                               "set_sparse_dim": target.get_num_classes(),
                               "size_base": "encoder",  # TODO: for RNA only...
                               "set_sparse": True}
  net["1_targetb_base"] = {"class": "copy",
                           "from": "existing_alignment",
                           "register_as_extern_data": new_target_name if train else None}
  # Framewise CE loss
  subnet["ce_loss"] = {
    "class": "activation",
    "from": output_log_prob_wb,
    "activation": "exp",
    "target": new_target_name,
    "loss": "ce",
    "loss_opts": {"focal_loss_factor": 2.0}
  }

  net["#config"] = {}
  # Update extern_data
  extern_data = get_global_config().get_of_type("extern_data", dict)
  _output_len_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="output-len")  # it's downsampled time
  extern_data["alignment"] = {"dim": target.get_num_classes(),
                              "sparse": True,
                              "same_dim_tags_as": {"t": _output_len_tag}}
  net["#config"]["extern_data"] = extern_data  # TODO: Why doesn't it work?

  # Change datasets to MetaDatasets
  def path_template(key):
    return f"{align_dir}/align.stage_{stage_num_align}_{key}.hdf"

  eval_datasets = get_global_config().get_of_type("eval_datasets", dict)
  net["#config"]["train"] = get_meta_dataset(train, "train", path_template("train"))
  net["#config"]["eval_datasets"] = {key: get_meta_dataset(train, key, path_template(key), True)
                                     for key in eval_datasets.keys()}
  return net


def get_meta_dataset(train: bool,
                     data_key: str,
                     path_2_hdf: str,
                     eval_ds: bool = False):
  """
  Creates the MetaDataset which combines:
  - align: FixedPath HdfDataset
  - default: Default Dataset
  The default may be LibriSpeechDataset, SwitchboardDataset, TimitDataset ..
  See for switchboard: https://github.com/rwth-i6/returnn-experiments/blob/master/2021-latent-monotonic-attention/switchboard/hard-att-local-win10-imp-recog.tnoextend96.ls01.laplace1000.hlr.config
  """
  # Default Dataset
  if eval_ds:
    default_dataset = get_global_config().get_of_type("eval_datasets", dict)[data_key]
  else:
    default_dataset = get_global_config().get_of_type(data_key, dict)
  assert default_dataset is not None, f"We couldn't find the {data_key} dataset in the base config."

  # FixedPath Dataset
  align_dataset = {
    "class": "HDFDataset", "files": [path_2_hdf],
    "use_cache_manager": True,
    "unique_seq_tags": True  # dev set can exist multiple times, needed??
  }

  # Options to overtake from the default dataset
  options = ["partition_epoch"]
  for opt in options:
    if opt in default_dataset:
      align_dataset[opt] = default_dataset[opt]
  # Options to overtake from the default dataset when training
  train_options = ["seq_ordering"]
  if train:
    for opt in options:
      if opt in train_options:
        align_dataset[opt] = default_dataset[opt]
  # TODO: used only for switchboard
  #   align_dataset["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"

  d = {
    "class": "MetaDataset",
    "datasets": {"default": default_dataset, "align": align_dataset},
    "data_map": {
      "data": ("default", "data"),
      # target: ("corpus", target),  #  needed for RNN-T chunking
      "alignment": ("align", "data"),
    },
    "seq_order_control_dataset": "default",  # it must support get_all_tags
  }
  return d
