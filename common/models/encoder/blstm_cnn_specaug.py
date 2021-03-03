
from returnn.import_ import import_
import numpy

import_("github.com/rwth-i6/returnn-experiments", "common")
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.asr.specaugment import specaugment


def make_net(src="data", *, num_lstm_layers=6, lstm_dim=1024, l2=0.0001, dropout=0.3, time_reduction=(3, 2)):
  orig_src = src
  net_dict = {
    "source": {"class": "eval", "eval": specaugment},
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
  if len(time_reduction) > num_lstm_layers - 1:
    time_reduction = [int(numpy.prod(time_reduction))]
  src = "conv_merged"
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": 1, "from": src},
      "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "direction": -1, "from": src}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": 1, "from": src,
        "L2": l2,  "dropout": dropout},
      "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "direction": -1, "from": src,
        "L2": l2, "dropout": dropout}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["output"] = {"class": "copy", "from": src}

  return {"class": "subnetwork", "subnetwork": net_dict, "from": orig_src}
