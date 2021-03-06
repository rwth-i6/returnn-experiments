
from __future__ import annotations
from typing import Dict, Any
from ..features import make_gt_features_opts
from .vocabs import bpe1k
from ...interface import DatasetConfig, VocabConfig

_Parts = [
  "train-clean-100", "train-clean-360", "train-other-500",
  "dev-clean", "dev-other",
  "test-clean", "test-other"]


class Librispeech(DatasetConfig):
  def __init__(self, audio_dim=50, train_epoch_split=20, vocab: VocabConfig = bpe1k):
    super(Librispeech, self).__init__()
    self.audio_dim = audio_dim
    self.train_epoch_split = train_epoch_split
    self.vocab = vocab

  def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
    return {
      "data": {"dim": self.audio_dim},
      "classes": {
        "sparse": True,
        "dim": self.vocab.get_num_classes(),
        "vocab": self.vocab.get_opts()},
    }

  def get_train_dataset(self) -> Dict[str, Any]:
    return self.get_dataset("train", train=True, train_partition_epoch=self.train_epoch_split)

  def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
    return {
      "dev": self.get_dataset("dev", train=False, subset=3000),
      "devtrain": self.get_dataset("train", train=False, subset=2000)}

  def get_dataset(self, key: str, *, train: bool, subset=None, train_partition_epoch=None):
    files = []
    parts = [part for part in _Parts if part.startswith(key)]
    assert parts
    for part in parts:
      files += ["data/dataset-ogg/%s.zip" % part, "data/dataset-ogg/%s.txt.gz" % part]
    d = {
      "class": 'OggZipDataset',
      "path": files,
      "zip_audio_files_have_name_as_prefix": False,
      "targets": self.vocab.get_opts(),
      "audio": {"norm_mean": "per_seq", "norm_std_dev": "per_seq", "num_feature_filters": self.audio_dim},
      # make_gt_features_opts(dim=self.audio_dim),
    }
    if train:
      d["partition_epoch"] = train_partition_epoch
      if key == "train":
        d["epoch_wise_filter"] = {
          (1, 5): {'max_mean_len': 200},
          (6, 10): {'max_mean_len': 500},
        }
      # d["audio"]["random_permute"] = True  # play around. note that this can be slow
      d["seq_ordering"] = "laplace:.1000"
    else:
      d["targets"]['unknown_label'] = '<unk>'  # only for non-train. for train, there never should be an unknown
      d["fixed_random_seed"] = 1
      d["seq_ordering"] = "sorted_reverse"
    if subset:
      d["fixed_random_subset"] = subset  # faster
    return d

