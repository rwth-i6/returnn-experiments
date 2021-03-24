
import sys
from returnn.config import get_global_config
from ..lm import Lm

config = get_global_config()


_lm_scale = 0.
if config.has("lm_scale"):
  _lm_scale = config.float("lm_scale", _lm_scale)
  print("** lm_scale %f" % _lm_scale, file=sys.stderr)

_lm_scale_internal = 0.1
if config.has("lm_scale_internal"):
  _lm_scale_internal = config.float("lm_scale_internal", _lm_scale)
  print("** lm_scale+internal %f" % _lm_scale_internal, file=sys.stderr)

_lm_eta = 1.5
if config.has("lm_eta"):
  _lm_eta = config.float("_lm_eta", _lm_eta)
  print("** lm_eta %f" % _lm_eta, file=sys.stderr)


_lm_fusion = "log-linear"
if config.has("lm_fusion"):
  _lm_fusion = config.value("lm_fusion", _lm_fusion)
  print("** lm_fusion %s" % _lm_fusion, file=sys.stderr)

_am_scale = None
if config.has("am_scale"):
  _am_scale = config.float("am_scale", _am_scale)
  print("** am_scale %f" % _am_scale, file=sys.stderr)


def eval_linear_interp(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  log1 = source(0)  # +log space, external_lm
  log2 = source(1)  # +log space, "internal" LM
  weighted_log1 = log1 + tf.math.log(_lm_scale)
  weighted_log2 = log2 + tf.math.log(1-_lm_scale)
  out = tf.math.reduce_logsumexp([weighted_log1, weighted_log2], axis=0)
  return out


def eval_log_linear_interp(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  log_external = source(0)  # external_lm
  log_internal = source(1)  # "internal" LM
  prev_u = source(2)
  prev_out_str = source(3)
  out = _lm_scale*log_external + (1-_lm_scale)*log_internal
  return out


def eval_local_fusion(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  log_external = source(0)  # external_lm
  log_internal = source(1)  # "internal" LM
  prev_u = source(2)
  prev_out_str = source(3)
  # We need 2 parameters here, because of the denominator (does not cancel)
  out = _lm_scale*log_external + _am_scale*log_internal
  return out - tf.math.reduce_logsumexp(out, axis=-1, keepdims=True)


def eval_log_linear(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  log_external = source(0)  # external_lm
  log_internal = source(1)  # "internal" LM
  prev_u = source(2)
  prev_out_str = source(3)
  out = _lm_scale*log_external + log_internal
  return out


def eval_density_ratio(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  from returnn.tf.util.basic import where_bc, safe_log, safe_exp
  log_external = source(0)  # external_lm
  log_internal = source(1)  # "internal" LM, label log-probs
  u = source(2)
  prev_out_str = source(3)
  log_zero_am = source(4)  # label log-probs with AM set to zero/mean.
  out = log_internal - _lm_scale_internal*log_zero_am + _lm_scale*log_external
  return out


def eval_density_ratio_interp(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  from returnn.tf.util.basic import where_bc, safe_log, safe_exp
  log_external = source(0)  # external_lm
  log_internal = source(1)  # "internal" LM, label log-probs
  u = source(2)
  prev_out_str = source(3)
  log_zero_am = source(4)  # label log-probs with AM set to zero/mean.
  out = (1-_lm_scale)*log_internal - _lm_scale_internal*log_zero_am + _lm_scale*log_external
  return out


def eval_lm_fusion(source, **kwargs):
  if _lm_fusion == "log-linear":
    print("Using log-linear LM combination, with LM-scale %.2f." % _lm_scale)
    return eval_log_linear(source, **kwargs)
  elif _lm_fusion == "linear-interp":
    print("Using linear LM combination, with LM-scale %.2f." % _lm_scale)
    return eval_linear_interp(source, **kwargs)
  elif _lm_fusion == "local-fusion":
    print("Using local fusion LM combination, with LM-scale %.2f and AM-scale %.2f." % (_lm_scale, _am_scale))
    return eval_local_fusion(source, **kwargs)
  elif _lm_fusion == "log-linear-interp":
    print("Using un-normalized log-linear LM interpolation, with LM-scale %.2f." % _lm_scale)
    return eval_log_linear_interp(source, **kwargs)
  elif _lm_fusion in ("divide-by-prior", "divide-by-prior-mean"):
    print("Using density ratio approach, with LM-scale %.2f" % (_lm_scale))
    return eval_density_ratio(source, **kwargs)
  elif _lm_fusion in ("divide-by-prior-interp", "divide-by-prior-mean-interp"):
    print("Using density ratio (interp) approach, with LM-scale %.2f and internal scale %.2f" % (_lm_scale, _lm_scale_internal))
    return eval_density_ratio_interp(source, **kwargs)
  else:
    raise TypeError(f"not supported {_lm_fusion!r}")


def make_internal_lm(*, am: str, lm: str):
  return {
    "class": "subnetwork", "from": am,
    "subnetwork": {
      "am_mean": {  # encoder mean
        "class": "reduce", "from": "data", "mode": "mean", "keep_dims": False, "axes": "t"},  # [B, enc-dim]

    }
  }
