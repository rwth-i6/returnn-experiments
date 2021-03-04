
from __future__ import annotations
from typing import Dict, Any
from returnn.tf.util.data import Data
from returnn.import_ import import_

import_("github.com/rwth-i6/returnn-experiments", "common")
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.encoder import blstm_cnn_specaug
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.recomb_recog import targetb_recomb_recog
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.loss import rnnt_loss
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.collect_out_str import make_out_str_func
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.datasets.interface import TargetConfig


def make_net(*, task: str, target: TargetConfig = None, beam_size: int = 12, l2=0.0001, lstm_dim=1024):
  if not target:
    target = TargetConfig()
  net_dict = {
    "encoder": blstm_cnn_specaug.make_encoder(l2=l2, lstm_dim=lstm_dim),
    "enc_seq_len": {"class": "length", "from": "encoder", "sparse": False},

    # for task "search" / search_output_layer
    "output_wo_b0": {
      "class": "masked_computation", "unit": {"class": "copy"},
      "from": "output", "mask": "output/output_emit"},
    "output_wo_b": {
      "class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target.vocab.get_num_classes()},
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target.key,
      'only_on_search': True},
  }

  if task == "train":
    net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % target.key}
    net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": 0}
    net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

  net_dict["output"] = make_decoder(
    train=(task == "train"), search=(task != "train"), l2=l2, target=target, beam_size=beam_size)

  return net_dict


def make_decoder(*, train: bool, search: bool, l2=0.0001, target: TargetConfig, beam_size: int = 12) -> Dict[str, Any]:
  """
  This is currently the original RNN-T label topology,
  meaning that we all vertical transitions in the lattice, i.e. U=T+S. (T input, S output, U alignment length).
  """
  # target is without blank.
  blank_idx = target.vocab.get_num_classes()
  return {
    "class": "rec",
    # In training, go framewise over the input, and inside the loop, we build up the whole 2D space (TxS).
    "from": [] if search else "encoder",
    "include_eos": True,
    "back_prop": train,
    "unit": {
      "am0": {"class": "gather_nd", "from": "base:encoder", "position": "prev:t"},  # [B,D]
      "am": {"class": "copy", "from": "am0" if search else "data:source"},

      "prev_out_non_blank": {
        "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": target.vocab.get_num_classes()},
      "lm_masked": {
        "class": "masked_computation",
        "mask": "prev:output_emit",
        "from": "prev_out_non_blank",  # in decoding
        "masked_from": None if search else "base:lm_input",  # enables optimization if used
        "unit": {
          "class": "subnetwork", "from": "data",
          "subnetwork": {
            "input_embed": {"class": "linear", "activation": None, "with_bias": False, "from": "data", "n_out": 256},
            "embed_dropout": {"class": "dropout", "from": "input_embed", "dropout": 0.2},
            # "lstm0": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "from": "embed_dropout", "L2": l2},
            "lstm0": {
              "class": "rnn_cell",
              "unit": "ZoneoutLSTM", "unit_opts": {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05},
              "from": "embed_dropout", "n_out": 500},
            "output": {"class": "copy", "from": "lstm0"}
          }
        }},
      "readout_in": {
        "class": "linear", "from": ["am", "lm_masked"],
        "activation": None, "n_out": 1000, "L2": l2, "dropout": 0.2,
        "out_type": {
          "batch_dim_axis": 0 if search else 2,
          "shape": (1000,) if search else (None, None, 1000),
          "time_dim_axis": None if search else 0}},  # (T, U+1, B, 1000)
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

      "label_log_prob": {
        "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3,
        "n_out": target.vocab.get_num_classes()},  # (B, T, U+1, 1030)
      "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1},  # (B, T, U+1, 1)
      "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
      "blank_log_prob": {
        "class": "eval", "from": "emit_prob0", "eval": "tf.compat.v1.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
      "label_emit_log_prob": {
        "class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1)
      "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]},  # (B, T, U+1, D+1)

      "full_sum_loss": {
        "class": "eval",
        "from": ["output_log_prob", "base:data:" + target.key, "base:encoder"],
        "eval": rnnt_loss,
        "out_type": lambda sources, **kwargs: Data(name="rnnt_loss", shape=()),
        "loss": "as_is",
      },

      "output": {
        "class": 'choice',
        'target': target.key,  # note: wrong! but this is ignored both in full-sum training and in search
        'beam_size': beam_size,
        'from': "output_log_prob", "input_type": "log_prob",
        "initial_output": 0,
        "length_normalization": False,
        "cheating": "exclusive" if train else None,  # only relevant for train+search
        "explicit_search_sources": ["prev:out_str", "prev:output"] if search and targetb_recomb_recog else None,
        "custom_score_combine": targetb_recomb_recog if search else None
      },
      # switchout only applicable to viterbi training, added below.
      "output_": {"class": "copy", "from": "output", "initial_output": 0},

      "out_str": {
        "class": "eval", "from": ["prev:out_str", "output_emit", "output"],
        "initial_output": None, "out_type": {"shape": (), "dtype": "string"},
        "eval": make_out_str_func(target=target.key)},

      "output_is_not_blank": {
        "class": "compare", "from": "output_", "value": blank_idx,
        "kind": "not_equal", "initial_output": True},

      # initial state=True so that we are consistent to the training and the initial state is correctly set.
      "output_emit": {"class": "copy", "from": "output_is_not_blank", "is_output_layer": True, "initial_output": True},

      "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},
      "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},

      # pos in target, [B]
      "du": {"class": "switch", "condition": "output_emit", "true_from": "const1", "false_from": "const0"},
      "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},

      # pos in input, [B]
      # output label: stay in t, otherwise advance t (encoder)
      "dt": {"class": "switch", "condition": "output_is_not_blank", "true_from": "const0", "false_from": "const1"},
      "t": {"class": "combine", "from": ["dt", "prev:t"], "kind": "add", "initial_output": 0},

      # stop at U+T
      # in recog: stop when all input has been consumed
      # in train: defined by target.
      "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater"},
    },
    "max_seq_len": "max_len_from('base:encoder') * 3",
  }