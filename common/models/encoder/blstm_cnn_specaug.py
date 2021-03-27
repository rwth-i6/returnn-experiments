
import numpy
from typing import Dict, Any, Union, Tuple

from ...asr.specaugment import specaugment_eval_func


def make_encoder(src="data", **kwargs):
  return {"class": "subnetwork", "subnetwork": make_net(**kwargs), "from": src}


def make_net(
    *,
    num_layers=6, lstm_dim=1024,
    time_reduction: Union[int, Tuple[int, ...]] = 6,
    with_specaugment=True,
    l2=0.0001, dropout=0.3, rec_weight_dropout=0.0,
) -> Dict[str, Any]:
  net_dict = {
    "source": {"class": "eval", "eval": specaugment_eval_func}
    if with_specaugment else {"class": "copy"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {"class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32,
              "activation": None, "with_bias": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {"class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32,
              "activation": None, "with_bias": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)
  }

  # Add encoder BLSTM stack.
  if isinstance(time_reduction, int):
    n = time_reduction
    time_reduction = []
    for i in range(2, n + 1):
      while n % i == 0:
        time_reduction.insert(0, i)
        n //= i
      if n <= 1:
        break
  assert isinstance(time_reduction, (tuple, list))
  while len(time_reduction) > num_layers - 1:
    time_reduction[:2] = [time_reduction[0] * time_reduction[1]]
  src = "conv_merged"
  opts = {"n_out": lstm_dim, "L2": l2}  # type: Dict[str, Any]
  if rec_weight_dropout:
    opts.setdefault("unit_opts", {})["rec_weight_dropout"] = rec_weight_dropout
  if num_layers >= 1:
    net_dict.update({
      "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "direction": 1, "from": src, **opts},
      "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "direction": -1, "from": src, **opts}})
    src = ["lstm0_fw", "lstm0_bw"]
  opts["dropout"] = dropout  # dropout (on input) only starting from lstm layer 2
  for i in range(1, num_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {"class": "rec", "unit": "nativelstm2", "direction": 1, "from": src, **opts},
      "lstm%i_bw" % i: {"class": "rec", "unit": "nativelstm2", "direction": -1, "from": src, **opts}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["output"] = {"class": "copy", "from": src}

  return net_dict
