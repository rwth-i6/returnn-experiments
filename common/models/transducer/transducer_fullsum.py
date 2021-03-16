
from __future__ import annotations
from typing import Dict, Any, Optional
from returnn.config import get_global_config

from ..encoder import blstm_cnn_specaug
from .. import lstm
from .recomb_recog import targetb_recomb_recog
from .loss import rnnt_loss, rnnt_loss_out_type
from ..collect_out_str import make_out_str_func
from ...datasets.interface import TargetConfig


def make_net(
    *,
    task: Optional[str] = None, target: TargetConfig = None,
    encoder_layer_dict: Dict[str, Any] = None,
    encoder_opts=None,
    decoder_opts=None,
    beam_size: int = 12,
    l2=0.0001,
    **enc_dec_flat_args
) -> Dict[str, Any]:
  if not task:
    task = get_global_config().value("task", "train")
  if not target:
    target = TargetConfig()
  encoder_opts = (encoder_opts or {}).copy()
  decoder_opts = (decoder_opts or {}).copy()
  for k, v in enc_dec_flat_args.items():
    if k.startswith("enc_"):
      encoder_opts[k[len("enc_"):]] = v
    elif k.startswith("dec_"):
      decoder_opts[k[len("dec_"):]] = v
    else:
      raise TypeError(f"unexpected argument {k}={v!r}")
  train = (task == "train")
  search = (task != "train")
  if not encoder_layer_dict:
    encoder_layer_dict = blstm_cnn_specaug.make_encoder(l2=l2, **encoder_opts)
  return {
    "encoder": encoder_layer_dict,
    "output": make_decoder(
      "encoder",
      train=train, search=search, l2=l2, target=target, beam_size=beam_size,
      **decoder_opts),

    # for task "search" / search_output_layer
    "output_wo_b": make_output_without_blank("output", target=target),
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target.key,
      'only_on_search': True},
  }


def make_decoder(
    encoder: str = "encoder",
    *,
    lm_layer_dict: Dict[str, Any] = None,
    lm_embed_dim=256,
    lm_dropout=0.2,
    lm_lstm_dim=512,
    readout_dropout=0.1,
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
  if not lm_layer_dict:
    lm_layer_dict = lstm.make_lstm(
      num_layers=1, lstm_dim=lm_lstm_dim,
      zoneout=True, embed_dropout=lm_dropout, embed_dim=lm_embed_dim)
  rec_decoder = {
    "am0": {"class": "gather_nd", "from": f"base:{encoder}", "position": "prev:t"},  # [B,D]
    "am": {"class": "copy", "from": "am0" if search else "data:source"},

    "prev_out_non_blank": {
      "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": target.get_num_classes()},
    # This is SlowRNN in the paper.
    "lm_masked": {
      "class": "masked_computation",
      "mask": "prev:output_emit",
      "from": "prev_out_non_blank",  # in decoding
      "masked_from": None if search else "lm_input",  # enables optimization if used
      "unit": lm_layer_dict
    },
    "lm": {"class": "copy", "from": "lm_masked"},
    "readout": make_readout(
      readout_dim=readout_dim, readout_dropout=readout_dropout,
      readout_l2=l2, output_dropout=output_dropout, target=target),

    "emit_prob0": {"class": "linear", "from": "readout/readout", "activation": None, "n_out": 1},  # (B, T, U+1, 1)
    "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
    "blank_log_prob": {
      "class": "eval", "from": "emit_prob0", "eval": "tf.compat.v1.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
    "label_emit_log_prob": {
      "class": "combine", "kind": "add", "from": ["readout/label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1)
    "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]},  # (B, T, U+1, D+1)

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

  if train:
    rec_decoder["full_sum_loss"] = {
      "class": "eval",
      "from": ["output_log_prob", f"base:data:{target.key}", f"base:{encoder}"],
      "eval": rnnt_loss,
      "out_type": rnnt_loss_out_type,
      "loss": "as_is",
    }

  if not search:
    rec_decoder["lm_input"] = {"class": "prefix_in_time", "from": f"base:data:{target.key}", "prefix": 0}

  return {
    "class": "rec",
    # In training, go framewise over the input, and inside the loop, we build up the whole 2D space (TxS).
    "from": [] if search else encoder,
    "include_eos": True,
    "back_prop": train,
    "unit": rec_decoder,
    "max_seq_len": f"max_len_from('base:{encoder}') * 3",
  }


def make_readout(
    *,
    shared: Optional[str] = None,
    am="am", lm="lm",
    readout_dim: int,
    readout_dropout: float,
    readout_l2: float,
    output_dropout: float,
    target: TargetConfig,
) -> Dict[str, Any]:
  shared_rel = f"base:{shared}/readout_lm" if shared else None
  net_dict = {
    "readout_lm": {
      "class": "linear", "from": f"base:{lm}",
      "activation": None, "n_out": readout_dim, "L2": readout_l2, "dropout": readout_dropout,
      "with_bias": True,
    } if not shared else {
      "class": "copy", "from": shared_rel
    },
    "readout_am": {
      "class": "linear", "from": "data",
      "activation": None, "n_out": readout_dim, "L2": readout_l2, "dropout": readout_dropout,
      "with_bias": False,  # only once, via readout_lm
      "reuse_params": shared_rel,
    },
    # Separate linear, such that this is more efficient with full-sum training.
    "readout_am_lm": {
      "class": "combine", "kind": "add", "from": ["readout_am", "readout_lm"],
      # (T, U+1, B, 1000) in search
      # "out_type": {
      #  "batch_dim_axis": 0 if search else 2,
      #  "shape": (readout_dim,) if search else (None, None, readout_dim),
      #  "time_dim_axis": None if search else 0}
    },  # (T, U+1, B, 1000)
    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_am_lm"},

    "label_log_prob": {
      "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": output_dropout,
      "n_out": target.get_num_classes(),
      "reuse_params": shared_rel},  # (B, T, U+1, 1030)

    "output": {"class": "copy", "from": "label_log_prob"}
  }

  return {"class": "subnetwork", "from": am, "subnetwork": net_dict}


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


def _safe_dict_update(d: Dict[str, Any], d2: Dict[str, Any]):
  for k, v in d2.items():
    assert k not in d
    d[k] = v
