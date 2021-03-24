

def make_lstm(source: str = "data", **kwargs):
  return {
    "class": "subnetwork", "from": source,
    "subnetwork": make_net(**kwargs)
  }


def make_net(
    source="data",
    *,
    num_layers=1, lstm_dim=512,
    dropout=0.0, l2=0.0,
    zoneout=False,
    embed=True, embed_dim=256, embed_with_bias=False,
    embed_dropout=0.0
):
  net_dict = {}
  if embed:
    net_dict["input_embed"] = {
      "class": "linear", "activation": None, "with_bias": embed_with_bias, "from": source, "n_out": embed_dim}
    source = "input_embed"
  opts = {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "L2": l2, "dropout": embed_dropout}
  if zoneout:
    opts.update({"unit": "ZoneoutLSTM", "unit_opts": {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05}})
  for i in range(num_layers):
    net_dict[f"lstm{i}"] = {"from": source, **opts}
    opts["dropout"] = dropout
    source = f"lstm{i}"
  net_dict["output"] = {"class": "copy", "from": source}
  return net_dict
