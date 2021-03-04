"""
This contains some of our standard vocabularies (e.g. BPE) for Librispeech.
"""

from pathlib import Path
from typing import Dict, Any
from ...interface import VocabConfig


class _Bpe(VocabConfig):
  def __init__(self, dim, codes: Path, vocab: Path):
    super(_Bpe, self).__init__()
    self.dim = dim
    self.codes = str(codes)
    self.vocab = str(vocab)

  def get_num_classes(self) -> int:
    return self.dim

  def get_opts(self) -> Dict[str, Any]:
    return {
      'bpe_file': self.codes,
      'vocab_file': self.vocab,
      'unknown_label': None  # should not be needed
    }


_vocab_dir = Path(__file__).absolute().parent / "vocabs"


bpe1k = _Bpe(
  dim=1056,
  codes=_vocab_dir / 'trans.bpe1k.codes',
  vocab=_vocab_dir / 'trans.bpe1k.vocab')


bpe10k = _Bpe(
  dim=10025,
  codes=_vocab_dir / 'trans.bpe10k.codes',
  vocab=_vocab_dir / 'trans.bpe10k.vocab')
