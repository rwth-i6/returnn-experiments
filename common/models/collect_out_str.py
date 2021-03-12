

from returnn.tf.layers.basic import EvalLayer
from returnn.datasets.generating import Vocabulary, BytePairEncoding


def get_vocab_tf(vocab: Vocabulary):
  from returnn.tf.util.basic import get_shared_vocab
  labels = list(vocab.labels)  # bpe labels ("@@" at end, or not), excluding blank
  if isinstance(vocab, BytePairEncoding):
    labels = [(label + " ").replace("@@ ", "") for label in labels]
  labels = labels + [""]  # maybe add extra blank (even if maybe not needed)
  labels_t = get_shared_vocab(labels)
  return labels_t


def get_vocab_sym(i, vocab: Vocabulary):
  """
  :param tf.Tensor i: e.g. [B], int32
  :param vocab:
  :return: same shape as input, string
  :rtype: tf.Tensor
  """
  from returnn.tf.compat import v1 as tf
  return tf.gather(params=get_vocab_tf(vocab), indices=i)


_out_str_func_cache = {}


def make_out_str_func(*, target: str):
  if target in _out_str_func_cache:
    return _out_str_func_cache[target]

  def out_str(self: EvalLayer, source, **_other):
    # sources: ["prev:out_str", "output_emit", "output"]
    from returnn.tf.compat import v1 as tf
    from returnn.tf.util.basic import where_bc
    target_data = self.network.extern_data.data[target]
    assert target_data.vocab
    return source(0) + where_bc(source(1), get_vocab_sym(source(2), vocab=target_data.vocab), tf.constant(""))

  _out_str_func_cache[target] = out_str
  return out_str
