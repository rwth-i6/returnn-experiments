__all__ = ['BuildCharacterVocab']

from sisyphus import *
import pickle

class BuildCharacterVocab(Job):
  """
  Build a character vocbulary for Returnn
  """

  def __init__(self, languages=["en"], uppercase=False):
    """

    :param list[str] languages:
    """
    self.languages = languages
    self.uppercase = uppercase
    self.out = self.output_path("orth_vocab.pkl")
    self.vocab_size = self.output_var("vocab_length")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    _pad = '_'
    _eos = '~'
    _space = ' '
    # ABCDEFGHIJKLMNÑOPQRSTUVWXYZÀÈÌÒÙÁÉÍÓÚ
    _characters = 'abcdefghijklmnñopqrstuvwxyzàèìòùáéíóúïü!\'"(),-.:;?'

    if 'es' in self.languages:
      _characters += '¿¡'
    if 'de' in self.languages:
      _characters += 'äöüß'
    if 'it' in self.languages:
      _characters += 'îû'
    if 'ca' in self.languages:
      _characters += 'ç'
    if 'fr' in self.languages:
      _characters += 'çæœÿêôë'

    # Export all symbols:
    # symbols = [_pad, _eos, _space] + list(l + '|' + c for l in hparams.languages for c in _characters)
    symbols = [_pad, _eos, _space] + list(c for c in _characters)
    if self.uppercase:
      symbols = [s.upper() for s in symbols]
    vocab = {k: v for v, k in enumerate(symbols)}
    pickle.dump(vocab, open(tk.uncached_path(self.out), "wb"))
    print("Vocab Size: %i" % len(symbols))
    self.vocab_size.set(len(symbols))
