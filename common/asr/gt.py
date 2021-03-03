
"""
Gammatone feature extraction.

Code by Peter Vieting, and adopted.
"""


import tensorflow as tf
import numpy
from returnn.tf.network import TFNetwork, ExternData

tf1 = tf.compat.v1


class Extractor:
  def __init__(self):
    net_dict = get_net_dict()
    with tf1.Graph().as_default() as self.graph:
      self.extern_data = ExternData({"data": {"shape": (None, 1)}})
      self.input = self.extern_data.data["data"]
      self.net = TFNetwork(name="Gammatone", extern_data=self.extern_data)
      self.net.construct_from_dict(net_dict)
      self.output = self.net.get_default_output_layer().output.copy_as_batch_major()
      self.session = tf1.Session(graph=self.graph)
      self.session.run(tf1.global_variables_initializer())

  def run(self, audio, seq_lens=None):
    """
    :param numpy.ndarray audio: shape [B,T,1] (T being raw samples, 16kHz)
    :param numpy.array|None seq_lens: shape [B]. if not given, assume [T]*B
    :return: features, feat_seq_lens
    :rtype: (numpy.ndarray,numpy.ndarray)
    """
    assert len(audio.shape) == 3 and audio.shape[-1] == 1
    b, t, _ = audio.shape
    if seq_lens is None:
      seq_lens = [t] * b
    return self.session.run(
      (self.output.placeholder, self.output.size_placeholder[0]),
      feed_dict={self.input.placeholder: audio, self.input.size_placeholder[0]: seq_lens})


_extractor = None


def _extract(*, audio, num_feature_filters, sample_rate, **_other):
  assert sample_rate == 16000
  assert num_feature_filters == 50
  global _extractor
  if not _extractor:
    _extractor = Extractor()
  features, _ = _extractor.run(audio[numpy.newaxis, :, numpy.newaxis])
  return features[0]


def make_returnn_audio_features_func():
  """
  This can be used for ExtractAudioFeatures in RETURNN,
  e.g. in OggZipDataset or LibriSpeechCorpus or others.
  """
  return _extract


def get_net_dict():
  return {
    'shift_0': {'class': 'slice', 'axis': 'T', 'slice_end': -1},
    'shift_1_raw': {'class': 'slice', 'axis': 'T', 'slice_start': 1},
    'shift_1': {'class': 'reinterpret_data', 'from': 'shift_1_raw', 'set_axes': {'T': 'time'}, 'size_base': 'shift_0'},
    'preemphasis': {'class': 'combine', 'from': ['shift_1', 'shift_0'], 'kind': 'sub'},
    'gammatone_filterbank_padding': {'class': 'pad', 'axes': 'T', 'from': 'preemphasis', 'padding': (574, 0)},
    'gammatone_filterbank': {
      'class': 'conv',
      'activation': 'abs',
      'filter_size': (640,),
      'forward_weights_init': {
        'class': 'GammatoneFilterbankInitializer', 'length': 0.04, 'num_channels': 50},
      'from': 'gammatone_filterbank_padding',
      'n_out': 50,
      'padding': 'valid'},
    'gammatone_filterbank_split': {'axis': 'F', 'class': 'split_dims', 'dims': (-1, 1), 'from': 'gammatone_filterbank'},
    'temporal_integration': {
      'class': 'conv',
      'filter_size': (400, 1),
      'forward_weights_init': 'numpy.hanning(400).reshape((400, 1, 1, 1))',
      'from': 'gammatone_filterbank_split',
      'n_out': 1,
      'padding': 'valid',
      'strides': (160, 1)},
    'temporal_integration_merge': {'class': 'merge_dims', 'axes': 'except_time', 'from': 'temporal_integration'},
    'compression': {
      'class': 'eval',
      'eval': 'tf.pow(source(0) + 1e-06, 0.1)',
      'from': 'temporal_integration_merge'},
    'dct': {'class': 'dct', 'from': 'compression'},
    'output': {
      'class': 'linear',
      'activation': None,
      'batch_norm': True,
      'forward_weights_init': 'Identity',
      'from': 'dct',
      'n_out': 50},
  }

