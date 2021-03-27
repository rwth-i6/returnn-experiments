
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from ..features import make_gt_features_opts
from .vocabs import bpe1k, bpe10k
from ...interface import DatasetConfig, VocabConfig
from ....data import get_common_data_path


_Parts = [
  "train-clean-100", "train-clean-360", "train-other-500",
  "dev-clean", "dev-other",
  "test-clean", "test-other"]

_norm_stats_dir = Path(__file__).absolute().parent / "norm_stats"


class Librispeech(DatasetConfig):
  def __init__(self, *,
               audio_dim=50,
               audio_norm: str = "per_seq",
               vocab: VocabConfig = bpe1k,
               train_epoch_split=20, train_random_permute=None):
    """
    :param audio_norm: "global" or "per_seq". "global" tries to read from standard location in repo
    """
    super(Librispeech, self).__init__()
    self.audio_dim = audio_dim
    self.audio_norm = audio_norm
    self.vocab = vocab
    self.train_epoch_split = train_epoch_split
    self.train_random_permute = train_random_permute

  @classmethod
  def old_defaults(cls, audio_dim=40, audio_norm="global", vocab: VocabConfig = bpe10k, **kwargs) -> Librispeech:
    return Librispeech(audio_dim=audio_dim, audio_norm=audio_norm, vocab=vocab, **kwargs)

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
      files += [
        # (History: Changed data/dataset-ogg -> data-common/librispeech/dataset/dataset-ogg)
        get_common_data_path("librispeech/dataset/dataset-ogg/%s.zip" % part),
        get_common_data_path("librispeech/dataset/dataset-ogg/%s.txt.gz" % part)]

    def _make_norm_arg(k: str):
      if self.audio_norm == "per_seq":
        return "per_seq"
      if self.audio_norm == "global":
        return str(_norm_stats_dir / f"stats.{self.audio_dim}.{k}.txt")
      if not self.audio_norm:
        return None
      raise TypeError(f"Invalid audio norm {self.audio_norm}.")

    d = {
      "class": 'OggZipDataset',
      "path": files,
      "use_cache_manager": True,
      "zip_audio_files_have_name_as_prefix": False,
      "targets": self.vocab.get_opts(),
      "audio": {
        "norm_mean": _make_norm_arg("mean"),
        "norm_std_dev": _make_norm_arg("std_dev"),
        "num_feature_filters": self.audio_dim},
      # make_gt_features_opts(dim=self.audio_dim),
    }  # type: Dict[str, Any]
    if train:
      d["partition_epoch"] = train_partition_epoch
      if key == "train":
        d["epoch_wise_filter"] = {
          (1, 5): {'max_mean_len': 200},
          (6, 10): {'max_mean_len': 500},
        }
      if self.train_random_permute:
        d["audio"]["random_permute"] = self.train_random_permute
      d["seq_ordering"] = "laplace:.1000"
    else:
      d["targets"]['unknown_label'] = '<unk>'  # only for non-train. for train, there never should be an unknown
      d["fixed_random_seed"] = 1
      d["seq_ordering"] = "sorted_reverse"
    if subset:
      d["fixed_random_subset"] = subset  # faster
    return d

