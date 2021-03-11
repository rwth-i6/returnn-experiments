
from __future__ import annotations
from typing import Dict, Any
from returnn.tf.util.data import Data

from ..encoder import blstm_cnn_specaug
from .recomb_recog import targetb_recomb_recog
from .loss import rnnt_loss
from ..collect_out_str import make_out_str_func
from ...datasets.interface import TargetConfig


def make_net(
    *,
    task: str, target: TargetConfig = None,
    encoder_layer_dict: Dict[str, Any] = None,
    encoder_opts=None,
    decoder_opts=None,
    beam_size: int = 12,
    l2=0.0001,
) -> Dict[str, Any]:
  if not target:
    target = TargetConfig()
  train = (task == "train")
  search = (task != "train")
  if not encoder_layer_dict:
    encoder_layer_dict = blstm_cnn_specaug.make_encoder(l2=l2, **(encoder_opts or {}))
  return {
    "encoder": encoder_layer_dict,
    "output": make_decoder(
      "encoder",
      train=train, search=search, l2=l2, target=target, beam_size=beam_size,
      **(decoder_opts or {})),

    # for task "search" / search_output_layer
    "output_wo_b": make_output_without_blank("output", target=target),
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target.key,
      'only_on_search': True},
  }


def make_decoder(
    encoder: str = "encoder",
    *,
    lm_embed_dim=256,
    lm_dropout=0.2,
    lm_lstm_dim=512,
    readout_dropout=0.2,
    readout_dim=1024,
    output_dropout=0.3,
    train: bool,
    search: bool, beam_size: int = 12,
    target: TargetConfig,
    l2=0.0001,
) -> Dict[str, Any]:
  """
  This is currently the original RNN-T label topology,
  meaning that we all vertical transitions in the lattice, i.e. U=T+S. (T input, S output, U alignment length).
  """
  blank_idx = target.get_num_classes()  # target is without blank.
  rec_decoder = {
    "am0": {"class": "gather_nd", "from": f"base:{encoder}", "position": "prev:t"},  # [B,D]
    "am": {"class": "copy", "from": "am0" if search else "data:source"},

    "prev_out_non_blank": {
      "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": target.get_num_classes()},
    "lm_masked": {
      "class": "masked_computation",
      "mask": "prev:output_emit",
      "from": "prev_out_non_blank",  # in decoding
      "masked_from": None if search else "lm_input",  # enables optimization if used
      "unit": {
        "class": "subnetwork", "from": "data",
        "subnetwork": {
          "input_embed": {
            "class": "linear", "activation": None, "with_bias": False, "from": "data", "n_out": lm_embed_dim},
          "embed_dropout": {"class": "dropout", "from": "input_embed", "dropout": lm_dropout},
          # "lstm0": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "from": "embed_dropout", "L2": l2},
          "lstm0": {
            "class": "rnn_cell",
            "unit": "ZoneoutLSTM", "unit_opts": {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05},
            "from": "embed_dropout", "n_out": lm_lstm_dim},
          "output": {"class": "copy", "from": "lstm0"}
        }
      }},
    "readout_in": {
      "class": "linear", "from": ["am", "lm_masked"],
      "activation": None, "n_out": readout_dim, "L2": l2, "dropout": readout_dropout,
      "out_type": {
        "batch_dim_axis": 0 if search else 2,
        "shape": (readout_dim,) if search else (None, None, readout_dim),
        "time_dim_axis": None if search else 0}},  # (T, U+1, B, 1000)
    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

    "label_log_prob": {
      "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": output_dropout,
      "n_out": target.get_num_classes()},  # (B, T, U+1, 1030)
    "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1},  # (B, T, U+1, 1)
    "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
    "blank_log_prob": {
      "class": "eval", "from": "emit_prob0", "eval": "tf.compat.v1.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
    "label_emit_log_prob": {
      "class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1)
    "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]},  # (B, T, U+1, D+1)

    "full_sum_loss": {
      "class": "eval",
      "from": ["output_log_prob", f"base:data:{target.key}", f"base:{encoder}"],
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
    "enc_seq_len": {"class": "length", "from": f"base:{encoder}", "sparse": False},
    "end": {"class": "compare", "from": ["t", "enc_seq_len"], "kind": "greater"},
  }

  if not search:
    rec_decoder.update({
      "lm_input0": {"class": "copy", "from": f"base:data:{target.key}"},
      "lm_input1": {"class": "prefix_in_time", "from": "lm_input0", "prefix": 0},
      "lm_input": {"class": "copy", "from": "lm_input1"},
    })

  return {
    "class": "rec",
    # In training, go framewise over the input, and inside the loop, we build up the whole 2D space (TxS).
    "from": [] if search else encoder,
    "include_eos": True,
    "back_prop": train,
    "unit": rec_decoder,
    "max_seq_len": f"max_len_from('base:{encoder}') * 3",
  }


def make_output_without_blank(decoder: str, *, target: TargetConfig = None):
  if not target:
    target = TargetConfig()
  return {
    "class": "subnetwork", "from": decoder, "subnetwork": {
      "output_wo_b0": {
        "class": "masked_computation", "unit": {"class": "copy"},
        "from": "data", "mask": f"base:{decoder}/output_emit"},
      "output": {
        "class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target.get_num_classes()},
    }
  }
