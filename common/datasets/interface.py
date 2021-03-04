
from __future__ import annotations
from typing import Dict, Optional


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

  def get_opts(self) -> Dict[str]:
    raise NotImplementedError


class TargetConfig:
  def __init__(self, vocab: VocabConfig, key: str):
    self.vocab = vocab
    self.key = key
