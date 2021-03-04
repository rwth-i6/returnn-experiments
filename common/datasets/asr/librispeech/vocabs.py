"""
This contains some of our standard vocabularies (e.g. BPE) for Librispeech.
"""

from typing import Dict
from ...interface import VocabConfig


class _Bpe(VocabConfig):
  def __init__(self, dim, codes, vocab):
    super(_Bpe, self).__init__()
    self.dim = dim
    self.codes = codes
    self.vocab = vocab

  def get_opts(self) -> Dict[str]:
    return {
      'bpe_file': self.codes,
      'vocab_file': self.vocab,
      'unknown_label': None  # should not be needed
    }


bpe1k = _Bpe(
  dim=1056,
  codes='data/dataset/trans.bpe1k.codes',
  vocab='data/dataset/trans.bpe1k.vocab')


bpe10k = _Bpe(
  dim=10025,
  codes='data/dataset/trans.bpe10k.codes',
  vocab='data/dataset/trans.bpe10k.vocab')
