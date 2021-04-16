"""
DummyDataset in RETURNN automatically downloads the data via `nltk`,
so no preparation is needed.
This is useful for demos/tests.
Note that this is only a subset of the official TIMIT corpus.
See :class:`NltkTimitDataset` for more details.
"""

from __future__ import annotations
from typing import Dict, Any
from returnn.config import get_global_config
from .librispeech.vocabs import bpe1k, bpe10k

from ..interface import DatasetConfig, VocabConfig


config = get_global_config()


class DummyDataset(DatasetConfig):
  def __init__(self, vocab: VocabConfig = bpe1k, audio_dim=50, seq_len=88, output_seq_len=8, num_seqs=32, debug_mode=None):
    super(DummyDataset, self).__init__()
    if debug_mode is None:
      debug_mode = config.typed_dict.get("debug_mode", False)
    self.audio_dim = audio_dim
    self.seq_len = seq_len
    self.output_seq_len = output_seq_len
    self.num_seqs = num_seqs
    self.vocab = vocab
    self.output_dim = vocab.get_num_classes()
    self.debug_mode = debug_mode

  def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
    return {
      "data": {"dim": self.audio_dim},
      "classes": {"sparse": True,
                  "dim": self.output_dim,
                  "vocab": self.vocab.get_opts()},
    }

  def get_train_dataset(self) -> Dict[str, Any]:
    return self.get_dataset("train")

  def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
    return {
      "dev": self.get_dataset("dev"),
      "devtrain": self.get_dataset("devtrain")}

  def get_dataset(self, key, subset=None):
    assert key in {"train", "devtrain", "dev"}
    print(f"Using {key} dataset!")
    return {
      "class": "DummyDatasetMultipleSequenceLength",
      "input_dim": self.audio_dim,
      "output_dim": self.output_dim,
      "seq_len": {
        'data': self.seq_len,
        'classes': self.output_seq_len
      },
      "num_seqs": self.num_seqs,
    }

