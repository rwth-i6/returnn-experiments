
"""
Model:

* Transducer, trained with full-sum, base model without external LM.
* Can be extended in :mod:`.lm_fusion` with external LM (and subtraction of internal LM).

Terminology / setting:

* label_wb: labels with blank
* label_nb: labels without blank
* Labels always include BOS/EOS, although transducer does not directly use it, but external LM might.
* In the interfaces for building blocks, not all inputs might be used.
* T: encoder num frames
* U: num frames of labels without blank without EOS/BOS

What buildings blocks / generic interfaces:

Generic interfaces:
* nn: dense_input -> dense_output (0/1 spatial dims)
* nn_rec: like nn but 1 spatial dim (auto-regressive) or optional implied recurrency (auto-regressive)
* nn_non_spat: like nn but ignoring spatial dims (e.g. linear)
* log_sm_out = linear . log_softmax (any spatial dims)
* lm = embed . nn . log_sm_out (0/1 spatial dims)

Generic building blocks:
* embed: sparse_labels -> dense (any spatial dims)
* linear (any spatial dims)
* log_softmax (any spatial dims)
* linear_combine: x1, ..., xn -> sum_i (linear|embed)_D(x_i) (i by name, not int)

Specific interface aliases:
* encoder = nn

* external_lm = lm (prev_sparse_label_nb (with BOS) -> label_nb_log_prob)

* decoder (to be run inside rec layer)
  * slow_rnn (lm) = nn_rec (prev_label_nb, prev_decoder_state, encoder -> slow_rnn_in -> dense_slow) (lstm)
    * slow_rnn_in = linear_combine (prev_label_nb, prev_decoder_state, encoder)
  * fast_rnn (s) = nn_rec (prev_label_wb, encoder (am), dense_slow (lm) -> fast_rnn_in -> dense_fast -> dense_readout)
    * fast_rnn_in = linear_combine|concat (prev_label_wb, encoder (am), dense_slow (lm))
    * readout = nn_non_spat (readout_in -> dense_readout) (might be 2D TxU)
      * readout_in = linear_combine (encoder, dense_fast, dense_slow)
  * log_prob_label_wb: dense_readout -> log_prob_label_wb
    * log_prob_label_nb:
    * log_prob_blank:


Principles I try to follow:

* Make building blocks. Avoid a complex mega class and derived subclasses.
* Generic interfaces are ok (i.e. a base class with abstract methods).

"""


from __future__ import annotations
from typing import Dict, Any, Optional, List
from returnn.config import get_global_config
import sys

from ..encoder import blstm_cnn_specaug
from .. import lstm
from .recomb_recog import targetb_recomb_recog
from .loss import rnnt_loss, rnnt_loss_out_type
from ..collect_out_str import make_out_str_func
from ...datasets.interface import TargetConfig


class Context:
  def __init__(self, *, task: Optional[str], beam_size: int, target: TargetConfig):
    self.task = task
    self.train = (task == "train")
    self.search = (task != "train")
    self.beam_size = beam_size
    self.target = target
    self.num_labels_nb = target.get_num_classes()
    self.num_labels_wb = self.num_labels_nb + 1
    self.blank_idx = self.num_labels_nb


LayerRef = str
LayerDict = Dict[str, Any]


class _IMaker:
  def __init__(self, *, ctx: Context):
    self.ctx = ctx

  def make(self, *args, **kwargs):
    raise NotImplementedError


class IEncoder(_IMaker):
  def make(self, source: LayerRef) -> LayerDict:
    raise NotImplementedError


class IDecoderSlowRnn(_IMaker):
  def make(self, *,
           prev_sparse_label_nb: LayerRef,
           prev_emit: LayerRef,
           unmasked_sparse_label_nb_seq: Optional[LayerRef] = None,
           prev_fast_rnn: LayerRef, encoder: LayerRef) -> LayerDict:
    raise NotImplementedError


class DecoderSlowRnnLstmIndependent(IDecoderSlowRnn):
  def __init__(self, num_layers=1, lstm_dim=512, zoneout=True, embed_dim=256, embed_dropout=0.2, **kwargs):
    super().__init__(**kwargs)
    self._lstm_opts = dict(
      num_layers=num_layers, lstm_dim=lstm_dim,
      zoneout=zoneout,
      embed_dim=embed_dim, embed_dropout=embed_dropout)

  def make(self, *,
           prev_sparse_label_nb: LayerRef,
           prev_emit: LayerRef,
           unmasked_sparse_label_nb_seq: Optional[LayerRef] = None,
           prev_fast_rnn: LayerRef, encoder: LayerRef) -> LayerDict:
    return make_masked(
      source=prev_sparse_label_nb,
      mask=prev_emit,
      masked_from=unmasked_sparse_label_nb_seq,  # enables optimization if used
      unmask=self.ctx.search,
      unit=lstm.make_lstm(**self._lstm_opts))


class IDecoderFastRnn(_IMaker):
  def make(self, *, prev_label_wb: LayerRef, encoder: LayerRef, slow_rnn: LayerRef) -> LayerDict:
    """
    prev_label_wb and encoder use the same time dim (T) (or none).
    slow_rnn can use the same (unmasked) (or none) or a different (U+1) (maybe in full-sum setting).
    When slow_rnn has a different time dim than prev_label_wb/encoder,
    we expect 2 time dims as output (Tx(U+1)).
    """
    raise NotImplementedError


class DecoderFastRnnOnlyReadout(IDecoderFastRnn):
  def __init__(self, *, readout_dim: int = 1024, readout_dropout=0.1, readout_l2=0.0001, **kwargs):
    super().__init__(**kwargs)
    self.readout_dim = readout_dim
    self.readout_dropout = readout_dropout
    self.readout_l2 = readout_l2

  def make(self, *, prev_label_wb: LayerRef, encoder: LayerRef, slow_rnn: LayerRef) -> LayerDict:
    return {
      "class": "subnetwork", "from": [],
      "subnetwork": {
        "in": LinearCombine(
          dim=self.readout_dim, dropout=self.readout_dropout, l2=self.readout_l2, ctx=self.ctx).make({
            # Note: Order is important here, because for multiple time dims, it will determine the order,
            # which is important for the RNN-T loss.
            "am": _base(encoder), "lm": _base(slow_rnn)}),
        "act": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "in"},
        "output": {"class": "copy", "from": "act"}
      }}


class IDecoderLogProbSeparateNb(_IMaker):
  def make(self, fast_rnn: LayerRef) -> LayerDict:
    raise NotImplementedError


class DecoderLogProbSeparateNb(IDecoderLogProbSeparateNb):
  def __init__(self, *, dropout=0.3, **kwargs):
    super().__init__(**kwargs)
    self.dropout = dropout

  def make(self, fast_rnn: LayerRef) -> LayerDict:
    return {
      "class": "linear", "activation": "log_softmax", "n_out": self.ctx.num_labels_nb,
      "from": fast_rnn, "dropout": self.dropout}


class IDecoderLogProbSeparateWb(_IMaker):
  def make(self, fast_rnn: LayerRef, log_prob_nb: LayerRef) -> LayerDict:
    raise NotImplementedError


class DecoderLogProbSeparateWb(IDecoderLogProbSeparateWb):
  def make(self, fast_rnn: LayerRef, log_prob_nb: LayerRef) -> LayerDict:
    return {
      "class": "subnetwork", "from": [],
      "subnetwork": {
        "emit_prob_in": {"class": "linear", "from": _base(fast_rnn), "activation": None, "n_out": 1},
        # (B, T, U+1, 1)
        "emit_log_prob": {
          "class": "activation", "from": "emit_prob_in", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
        "blank_log_prob": {
          "class": "eval", "from": "emit_prob_in", "eval": "tf.compat.v1.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)

        "label_emit_log_prob": {
          "class": "combine", "kind": "add", "from": [_base(log_prob_nb), "emit_log_prob"]},
        # (B, T, U+1, 1), scaling factor in log-space
        "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]},  # (B, T, U+1, N)
        "output": {"class": "copy", "from": "output_log_prob", "n_out": self.ctx.num_labels_wb}
      }
    }


class Decoder(_IMaker):
  def __init__(self, *,
               slow_rnn: IDecoderSlowRnn = None,
               fast_rnn: IDecoderFastRnn = None,
               log_prob_separate_nb: IDecoderLogProbSeparateNb = None,
               log_prob_separate_wb: IDecoderLogProbSeparateWb = None,
               **kwargs):
    super().__init__(**kwargs)
    if not slow_rnn:
      slow_rnn = DecoderSlowRnnLstmIndependent(ctx=self.ctx)
    self.slow_rnn = slow_rnn
    if not fast_rnn:
      fast_rnn = DecoderFastRnnOnlyReadout(ctx=self.ctx)
    self.fast_rnn = fast_rnn
    if not log_prob_separate_nb:
      log_prob_separate_nb = DecoderLogProbSeparateNb(ctx=self.ctx)
    self.log_prob_separate_nb = log_prob_separate_nb
    if not log_prob_separate_wb:
      log_prob_separate_wb = DecoderLogProbSeparateWb(ctx=self.ctx)
    self.log_prob_separate_wb = log_prob_separate_wb

  def make(self, encoder: LayerRef):
    target = self.ctx.target
    beam_size = self.ctx.beam_size
    train = self.ctx.train
    search = self.ctx.search
    blank_idx = self.ctx.blank_idx

    rec_decoder = {
      "am0": {"class": "gather_nd", "from": _base(encoder), "position": "prev:t"},  # [B,D]
      "am": {"class": "copy", "from": "am0" if search else "data:source"},

      "prev_out_non_blank": {
        "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": target.get_num_classes()},

      "slow_rnn": self.slow_rnn.make(
        prev_sparse_label_nb="prev_out_non_blank",
        prev_emit="prev:output_emit",
        unmasked_sparse_label_nb_seq=None if search else "lm_input",  # might enable optimization if used
        prev_fast_rnn="prev:fast_rnn",
        encoder="am"),

      "fast_rnn": self.fast_rnn.make(
        prev_label_wb="prev:output_",
        slow_rnn="slow_rnn",
        encoder="am"),

      "output_log_prob_nb": self.log_prob_separate_nb.make(fast_rnn="fast_rnn"),
      "output_log_prob_wb": self.log_prob_separate_wb.make(fast_rnn="fast_rnn", log_prob_nb="output_log_prob_nb"),

      "output": {
        "class": 'choice',
        'target': target.key,  # note: wrong! but this is ignored both in full-sum training and in search
        'beam_size': beam_size,
        'from': "output_log_prob_wb", "input_type": "log_prob",
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
        "from": ["output_log_prob_wb", f"base:data:{target.key}", f"base:{encoder}"],
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


class EncoderBLstmCnnSpecAug(IEncoder):
  def __init__(self,
               num_layers=6, lstm_dim=1024,
               time_reduction=(3, 2), with_specaugment=True,
               l2=0.0001, dropout=0.3, rec_weight_dropout=0.0,
               **kwargs):
    super().__init__(**kwargs)
    self._encoder_opts = dict(
      num_layers=num_layers, lstm_dim=lstm_dim,
      time_reduction=time_reduction, with_specaugment=with_specaugment,
      l2=l2, dropout=dropout, rec_weight_dropout=rec_weight_dropout)

  def make(self, source: LayerRef) -> LayerDict:
    return blstm_cnn_specaug.make_encoder(source, **self._encoder_opts)


class Net:
  def __init__(self, *, ctx: Context, encoder: IEncoder = None, decoder: Decoder = None, aux_ctc_loss=False):
    self.ctx = ctx
    if not encoder:
      encoder = EncoderBLstmCnnSpecAug(ctx=ctx)
    self.encoder = encoder
    if not decoder:
      decoder = Decoder(ctx=ctx)
    self.decoder = decoder
    self.aux_ctc_loss = aux_ctc_loss

  def make_net(self) -> Dict[str, Any]:
    net = {
      "encoder": self.encoder.make("data"),
      "output": self.decoder.make("encoder"),

      # for task "search" / search_output_layer
      "output_wo_b": make_output_without_blank("output", target=self.ctx.target),
      "decision": {
        "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": self.ctx.target.key,
        'only_on_search': True},
    }
    if self.aux_ctc_loss:
      net["ctc"] = make_ctc_loss("encoder", target=self.ctx.target)
    return net


def make_net(
    *,
    task: Optional[str] = None, target: TargetConfig = None,
    encoder_opts=None,
    decoder_opts=None,
    beam_size: int = 12,
    l2=0.0001,
    **kwargs
) -> Dict[str, Any]:
  if not task:
    task = get_global_config().value("task", "train")
  if not target:
    target = TargetConfig.global_from_config()
  encoder_opts = (encoder_opts or {}).copy()
  decoder_opts = (decoder_opts or {}).copy()
  for k in list(kwargs.keys()):
    if k.startswith("enc_"):
      encoder_opts[k[len("enc_"):]] = kwargs.pop(k)
    elif k.startswith("dec_"):
      decoder_opts[k[len("dec_"):]] = kwargs.pop(k)
  encoder_opts.setdefault("l2", l2)
  decoder_opts.setdefault("l2", l2)
  ctx = Context(task=task, beam_size=beam_size, target=target)
  net = Net(
    ctx=ctx,
    encoder=EncoderBLstmCnnSpecAug(ctx=ctx, **encoder_opts),
    decoder=_make_decoder(ctx=ctx, **decoder_opts),
    **kwargs)
  return net.make_net()


def _make_decoder(
    *,
    ctx: Context,
    lm_embed_dim=256,
    lm_dropout=0.2,
    lm_lstm_dim=512,
    readout_dropout=0.1,
    readout_dim=1024,
    output_dropout=0.3,
    l2=0.0001,
) -> Decoder:
  """
  This is currently the original RNN-T label topology,
  meaning that we all vertical transitions in the lattice, i.e. U=T+S. (T input, S output, U alignment length).
  """
  return Decoder(
    ctx=ctx,
    slow_rnn=DecoderSlowRnnLstmIndependent(
      ctx=ctx, lstm_dim=lm_lstm_dim, embed_dim=lm_embed_dim, embed_dropout=lm_dropout),
    fast_rnn=DecoderFastRnnOnlyReadout(
      ctx=ctx, readout_dim=readout_dim, readout_dropout=readout_dropout, readout_l2=l2),
    log_prob_separate_nb=DecoderLogProbSeparateNb(ctx=ctx, dropout=output_dropout),
    log_prob_separate_wb=DecoderLogProbSeparateWb(ctx=ctx))


def make_ctc_loss(src: LayerRef, *, target: TargetConfig):
  return {
    "class": "softmax", "from": src, "loss": "ctc", "target": target.key,
    "loss_opts": {"beam_width": 1, "use_native": True}}


def make_output_without_blank(decoder: str, *, target: TargetConfig = None):
  if not target:
    target = TargetConfig.global_from_config()
  return {
    "class": "subnetwork", "from": decoder, "subnetwork": {
      "output_wo_b0": {
        "class": "masked_computation", "unit": {"class": "copy"},
        "from": "data", "mask": f"base:{decoder}/output_emit"},
      "output": {
        "class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target.get_num_classes()},
    }
  }


def make_masked(
      source: str, *,
      mask: str,
      masked_from: Optional[str] = None,
      unmask: bool = True,
      unit: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "class": "subnetwork", "from": [], "subnetwork": {
      "masked": {
        "class": "masked_computation",
        "mask": f"base:{mask}",
        "masked_from": f"base:{masked_from}" if masked_from else None,
        "from": f"base:{source}",
        "unit": unit
      },
      "unmask": {"class": "unmask", "from": "masked", "mask": f"base:{mask}"},
      "output": {"class": "copy", "from": "unmask" if unmask else "masked"}
    }
  }


class LinearCombine(_IMaker):
  def __init__(self, *, dim: int, dropout: float = 0, l2: float = 0, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.dropout = dropout
    self.l2 = l2

  def make(self, inputs: Dict[str, LayerRef]) -> LayerDict:
    return {
      "class": "subnetwork", "from": [],
      "subnetwork": {
        "bias": {"class": "variable", "add_batch_axis": False, "shape": [self.dim]},
        **{
          f"lin_{name}": {
            "class": "linear", "activation": None, "with_bias": False, "n_out": self.dim,
            "from": _base(layer),
            "dropout": self.dropout, "L2": self.l2}
          for name, layer in inputs.items()
        },
        "output": {
          "class": "combine", "kind": "add",
          "from": ["bias"] + [f"lin_{name}" for name in _det_keys(inputs)]}
      }
    }


def _base(name: LayerRef) -> LayerRef:
  return f"base:{name}"


def _det_keys(d: Dict[str, Any]) -> List[str]:
  assert sys.version_info[:2] >= (3, 6)  # the order might have influence on the behavior
  return list(d.keys())  # insertion-order is kept


def _safe_dict_update(d: Dict[str, Any], d2: Dict[str, Any]):
  for k, v in d2.items():
    assert k not in d
    d[k] = v
