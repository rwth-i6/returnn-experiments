
from __future__ import annotations
from typing import Dict, Optional, Any


class DatasetConfig:
  def get_extern_data(self) -> Dict[str, Dict[str]]:
    raise NotImplementedError

  def get_train_dataset(self) -> Dict[str]:
    raise NotImplementedError

  def get_eval_datasets(self) -> Dict[str, Dict[str]]:
    """
    :return: e.g. {"dev": ..., "devtrain": ...}
    """
    raise NotImplementedError

  # noinspection PyMethodMayBeStatic
  def get_default_input(self) -> Optional[str]:
    """
    What is the default input data key of the dataset.
    (If that is meaningful.) (Must be in extern data.)
    """
    return "data"

  # noinspection PyMethodMayBeStatic
  def get_default_target(self) -> Optional[str]:
    """
    What is the default target key of the dataset.
    (If that is meaningful.) (Must be in extern data.)
    """
    return "classes"

  def get_config_opts(self) -> Dict[str]:
    """
    E.g. in your main config, you could do::

      globals().update(dataset.get_config_opts())
    """
    return {
      "extern_data": self.get_extern_data(),
      "train": self.get_train_dataset(),
      "eval_datasets": self.get_eval_datasets(),
      "target": self.get_default_target()}


class VocabConfig:
  def get_num_classes(self) -> int:
    raise NotImplementedError

  def get_opts(self) -> Dict[str, Any]:
    raise NotImplementedError


class VocabConfigStatic(VocabConfig):
  def __init__(self, *, num_classes: int, opts: Dict[str, Any]):
    super(VocabConfigStatic, self).__init__()
    self.num_classes = num_classes
    self.opts = opts

  @classmethod
  def from_global_config(cls, data_key: str):
    from returnn.config import get_global_config
    config = get_global_config()
    extern_data_opts = config.typed_dict["extern_data"]
    data_opts = extern_data_opts[data_key]
    return VocabConfigStatic(num_classes=data_opts["dim"], opts=data_opts["vocab"])

  def get_num_classes(self) -> int:
    return self.num_classes

  def get_opts(self) -> Dict[str, Any]:
    return self.opts


class TargetConfig:
  def __init__(self, key: str = None, *, vocab: VocabConfig = None):
    if not key:
      from returnn.config import get_global_config
      config = get_global_config()
      key = config.typed_dict["target"]
    if not vocab:
      vocab = VocabConfigStatic.from_global_config(key)
    self.vocab = vocab
    self.key = key
