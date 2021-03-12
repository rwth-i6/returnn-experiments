
from __future__ import annotations
from typing import Dict, Tuple, Union, Any, Optional
from returnn.config import get_global_config


_Num = Union[int, float]


class Pretrain:
  """
  Features:

  * Can grow the network.
  * Learning rate warmup.
  * Disables regularization initially.

  In your config, do sth like::

    get_network = Pretrain(...).get_network
  """

  def __init__(self, make_net, make_net_args: Optional[Dict[str, Tuple[_Num, _Num]]] = None, num_epochs: int = 10):
    self._is_initialized = False  # we lazily init late, to make sure all config opts are set (e.g. learning_rate)
    self._make_net = make_net
    self._make_net_args = make_net_args or {}
    self._lr_warmup_num_epochs = num_epochs // 2
    self._pretrain_num_epochs = num_epochs

  def get_network(self, epoch: int, **_kwargs) -> Dict[str, Any]:
    self._lazy_init()
    epoch0 = epoch - 1  # RETURNN starts with epoch 1, but 0-indexed is easier here

    def resolve(arg, **kwargs):
      return self._resolve_make_net_arg(epoch0, arg, **kwargs)

    make_net_args = {k: resolve(v) for (k, v) in self._make_net_args.items()}
    net_dict = self._make_net(**make_net_args)
    assert isinstance(net_dict, dict)
    net_dict = net_dict.copy()

    if epoch0 < self._pretrain_num_epochs:
      config_ = net_dict.get("#config", {})

      config_["learning_rate"] = resolve(
        (self._lr_warmup_initial, self._lr_std), num_epochs=self._lr_warmup_num_epochs)

      if config_:
        net_dict["#config"] = config_

      net_dict["#copy_param_mode"] = "subset"

    return net_dict

  def _lazy_init(self):
    if self._is_initialized:
      return
    self._is_initialized = True

    config = get_global_config()
    self._lr_std = config.typed_dict["learning_rate"]
    self._lr_warmup_initial = self._lr_std / 10.

  def _resolve_make_net_arg(self, epoch0: int, arg: Tuple[_Num, _Num], num_epochs: Optional[int] = None):
    start, end = arg
    if num_epochs is None:
      num_epochs = self._pretrain_num_epochs
    if epoch0 >= num_epochs:
      return end
    f = epoch0 / (float(num_epochs) - 1)  # 0..1
    res = end * f + start * (1. - f)
    return type(end)(res)
