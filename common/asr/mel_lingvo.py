"""
We borrow some code from `Lingvo <https://github.com/tensorflow/lingvo/>`__.
Specifically the `ASR frontend <https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/frontend.py>`__,
commit db58ae9.
Also see `create_asr_features <https://github.com/tensorflow/lingvo/blob/master/lingvo/tools/create_asr_features.py>`__.
"""

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
import collections
import math
from typing import Dict, Optional
from returnn.tf.util.basic import get_shape
import tensorflow as tf


class _ReturnnAudioFeatureExtractor:
  def __init__(self):
    self.opts = None  # type: Optional[Params]
    self.tf_session = None  # type: Optional[tf.compat.v1.Session]
    self.tf_audio_placeholder = None
    self.tf_log_mel = None

  def __repr__(self):
    return f"{self.__class__.__name__}{self.opts.items() if self.opts else 'uninitialized'}"

  def setup(self, opts: Params):
    if self.opts:
      self.opts.assert_same(opts)
      return
    self.opts = opts

    with tf.compat.v1.Graph().as_default() as g:
      self.tf_audio_placeholder = tf.compat.v1.placeholder(name="audio", dtype=tf.float32, shape=(None,))
      log_mel = extract_log_mel_features_from_audio(self.tf_audio_placeholder, **opts.items())
      log_mel = tf.squeeze(log_mel, axis=0)
      log_mel = tf.squeeze(log_mel, axis=-1)
      self.tf_log_mel = log_mel
      self.tf_session = tf.compat.v1.Session(graph=g)

  def extract(self, *, audio, num_feature_filters, sample_rate, **_other):
    assert sample_rate == self.opts.sample_rate
    assert num_feature_filters == self.opts.num_bins
    return self.tf_session.run(self.tf_log_mel, feed_dict={self.tf_audio_placeholder: audio})


_global_returnn_audio_feature_extractor = _ReturnnAudioFeatureExtractor()


def make_returnn_audio_features_func(cached=_global_returnn_audio_feature_extractor, **opts):
  """
  This can be used for ExtractAudioFeatures in RETURNN,
  e.g. in OggZipDataset or LibriSpeechCorpus or others.

  This is cached by default, but you can disable this by setting cached=None.
  """
  if cached:
    extractor = cached
  else:
    extractor = _ReturnnAudioFeatureExtractor()
  p = default_asr_frontend_params()
  p.update(opts)
  extractor.setup(p)
  return extractor.extract


def default_asr_frontend_params(num_bins=80, sample_rate=16000):
  p = MelAsrFrontend.make_params()
  p.sample_rate = float(sample_rate)
  p.frame_size_ms = 25.
  p.frame_step_ms = 10.
  p.num_bins = num_bins
  p.lower_edge_hertz = 125.
  p.upper_edge_hertz = 7600.
  p.preemph = 0.97
  p.noise_scale = 0.
  p.pad_end = False
  return p


def extract_log_mel_features_from_audio(audio, **opts):
  """Create Log-Mel Filterbank Features from audio samples.
  Args:
    audio: Tensor representing audio samples (normalized in [-1,1)).
      It is currently assumed that the wav file is encoded at 16KHz.
      Shape [num_frames] or [num_frames,1].
  Returns:
    A Tensor representing three stacked log-Mel filterbank energies, sub-sampled
    every three frames.
  """
  assert isinstance(audio, tf.Tensor)
  audio *= 32768  # that's what it expects...
  if audio.shape.ndims >= 2:
    # Remove channel dimension, since we have a single channel.
    audio = tf.squeeze(audio, axis=1)
  audio = tf.expand_dims(audio, axis=0)  # [B,num_frames]
  p = default_asr_frontend_params()
  p.update(opts)
  mel_frontend = MelAsrFrontend(**p.items())
  log_mel, _ = mel_frontend.fprop(audio)
  return log_mel


def extract_log_mel_features_from_wav(wav_bytes_t, **opts):
  """Create Log-Mel Filterbank Features from raw bytes.
  Args:
    wav_bytes_t: Tensor representing raw wav file as a string of bytes. It is
      currently assumed that the wav file is encoded at 16KHz (see DecodeWav,
      below).
  Returns:
    A Tensor representing three stacked log-Mel filterbank energies, sub-sampled
    every three frames.
  """
  result = tf.audio.decode_wav(wav_bytes_t)
  sample_rate, audio = result.sample_rate, result.audio
  audio *= 32768
  # Remove channel dimension, since we have a single channel.
  audio = tf.squeeze(audio, axis=1)
  # TODO(drpng): make batches.
  audio = tf.expand_dims(audio, axis=0)
  p = default_asr_frontend_params()
  p.update(opts)
  mel_frontend = MelAsrFrontend(**p.items())
  with tf.control_dependencies(
        [tf.assert_equal(sample_rate, int(p.sample_rate))]):
    log_mel, _ = mel_frontend.fprop(audio)
  return log_mel


def _next_power_of_two(i):
  return math.pow(2, math.ceil(math.log(i, 2)))


class MelAsrFrontend:
  """An AsrFrontend that implements mel feature extraction from PCM frames.
  This is expressed in pure TensorFlow and without reference to external
  resources.
  The frontend implements the following stages:
      `Framer -> Window -> FFT -> FilterBank -> MeanStdDev -> SubSample`
  The FProp input to this layer can either have rank 3 or rank 4 shape:
      [batch_size, timestep, packet_size, channel_count]
      [batch_size, timestep * packet_size, channel_count]
  For compatibility with existing code, 2D [batch_size, timestep] mono shapes
  are also supported.
  In the common case, the packet_size is 1. The 4D variant is accepted for
  glueless interface to input generators that frame their input samples in
  some way. The external framing choice does not influence the operation of
  this instance, but it is accepted.
  TODO(laurenzo): Refactor call sites to uniformly use the 4D variant and
  eliminate fallback logic in this class.
  Only 1 channel is currently supported.
  TODO(laurenzo): Refactor this class to operate on multi-channel inputs.
  """

  @classmethod
  def make_params(cls):
    p = Params()
    p.name = 'frontend'
    p.define("random_seed", 0)
    p.define('sample_rate', 16000.0, 'Sample rate in Hz')
    p.define('channel_count', 1, 'Number of channels.')
    p.define('frame_size_ms', 25.0,
             'Amount of data grabbed for each frame during analysis')
    p.define('frame_step_ms', 10.0, 'Number of ms to jump between frames')
    p.define('num_bins', 80, 'Number of bins in the mel-spectrogram output')
    p.define('lower_edge_hertz', 125.0,
             'The lowest frequency of the mel-spectrogram analsis')
    p.define('upper_edge_hertz', 7600.0,
             'The highest frequency of the mel-spectrogram analsis')
    p.define(
        'preemph', 0.97,
        'The first-order filter coefficient used for preemphasis. When it '
        'is 0.0, preemphasis is turned off.')
    p.define('noise_scale', 8.0,
             'The amount of noise (in 16-bit LSB units) to add')
    p.define('window_fn', 'HANNING',
             'Window function to apply (valid values are "HANNING", and None)')
    p.define(
        'pad_end', False,
        'Whether to pad the end of `signals` with zeros when the provided '
        'frame length and step produces a frame that lies partially past '
        'its end.')
    p.define(
        'per_bin_mean', None,
        'Per-bin (num_bins) means for normalizing the spectrograms. '
        'Defaults to zeros.')
    p.define('per_bin_stddev', None,
             'Per-bin (num_bins) standard deviations. Defaults to ones.')
    p.define('stack_left_context', 0, 'Number of left context frames to stack.')
    p.define('stack_right_context', 0,
             'Number of right context frames to stack.')
    p.define('frame_stride', 1, 'The frame stride for sub-sampling.')

    p.define('fft_overdrive', True,
             'Whether to use twice the minimum fft resolution.')
    p.define('output_floor', 1.0,
             'Minimum output of filterbank output prior to taking logarithm.')
    p.define(
        'compute_energy', False,
        'Whether to compute filterbank output on the energy of spectrum '
        'rather than just the magnitude.')
    p.define('use_divide_stream', False,
             'Whether use a divide stream to the input signal.')
    return p

  def __init__(self, **kwargs):
    """
    For allowed kwargs, see self._make_params.
    """
    self.params = self.make_params()
    self.params.update(kwargs)
    p = self.params
    if p.frame_stride < 1:
      raise ValueError('frame_stride must be positive.')

    assert p.channel_count == 1, 'Only 1 channel currently supported.'
    # Make sure key params are in floating point.
    p.sample_rate = float(p.sample_rate)
    p.frame_step_ms = float(p.frame_step_ms)
    p.frame_size_ms = float(p.frame_size_ms)
    p.lower_edge_hertz = float(p.lower_edge_hertz)
    p.upper_edge_hertz = float(p.upper_edge_hertz)

    self._frame_step = int(round(p.sample_rate * p.frame_step_ms / 1000.0))
    self._frame_size = (int(round(p.sample_rate * p.frame_size_ms / 1000.0)) + 1)  # +1 for the preemph

    self._fft_size = int(max(512, _next_power_of_two(self._frame_size)))
    if p.fft_overdrive:
      self._fft_size *= 2

    self._create_window_function()

    # Mean/stddev.
    if p.per_bin_mean is None:
      p.per_bin_mean = [0.0] * p.num_bins
    if p.per_bin_stddev is None:
      p.per_bin_stddev = [1.0] * p.num_bins
    assert len(p.per_bin_mean) == p.num_bins
    assert len(p.per_bin_stddev) == p.num_bins

  def _create_window_function(self):
    p = self.params
    if p.window_fn is None:
      self._window_fn = None
    elif p.window_fn == 'HANNING':

      def _hanning_window(frame_size, dtype):
        return tf.signal.hann_window(frame_size, dtype=dtype)

      self._window_fn = _hanning_window
    else:
      raise ValueError('Illegal value %r for window_fn param' % p.window_fn)

  @property
  def window_frame_size(self):
    return self._frame_size

  @property
  def window_frame_step(self):
    return self._frame_step

  @staticmethod
  def _remove_channel_dim(pcm_audio_data):
    if pcm_audio_data.shape.rank == 3:
      pcm_audio_data = tf.squeeze(pcm_audio_data, 2)
      assert pcm_audio_data.shape.rank == 2, (
          'MelAsrFrontend only supports one channel')
    return pcm_audio_data

  @staticmethod
  def _reshape_to_mono2d(pcm_audio_data, paddings):
    """Reshapes a 3D or 4D input to 2D.
    Since the input to FProp can be 3D or 4D (see class comments), this will
    collapse it back to a 2D, mono shape for internal processing.
    Args:
      pcm_audio_data: 2D, 3D or 4D audio input. See class comments. Must have a
        rank.
      paddings: Original paddings shaped to the first two dims of
        pcm_audio_data.
    Returns:
      Tuple of 2D [batch_size, timestep] mono audio data, new paddings.
    """
    shape = get_shape(pcm_audio_data)
    rank = len(shape)
    if rank == 2:
      return pcm_audio_data, paddings
    elif rank == 3:
      # [batch, time, channel]
      with tf.control_dependencies([tf.assert_equal(shape[2], 1)]):
        return tf.squeeze(pcm_audio_data, axis=2), paddings
    elif rank == 4:
      # [batch, time, packet, channel]
      batch_size, orig_time, orig_packet_size, channel = shape
      time = orig_time * orig_packet_size
      with tf.control_dependencies([tf.assert_equal(channel, 1)]):
        pcm_audio_data = tf.reshape(pcm_audio_data, (batch_size, time))
        # Transform paddings into the new time base with a padding per time
        # step vs per packet by duplicating each packet.
        paddings = tf.reshape(
            tf.tile(tf.expand_dims(paddings, axis=2), [1, 1, orig_packet_size]),
            (batch_size, time))
        return pcm_audio_data, paddings
    else:
      raise ValueError('Illegal pcm_audio_data shape')

  def fprop(self, audio, paddings=None):
    """Perform signal processing on a sequence of PCM data.
    NOTE: This implementation does not currently support paddings, and they
    are accepted for compatibility with the super-class.
    TODO(laurenzo): Rework this to support paddings.
    Args:
      audio: int16 or float32 tensor of PCM audio data, scaled to
        [-32768..32768] (versus [-1..1)!). See class comments for supported
        input shapes.
      paddings: per frame 0/1 paddings. Shaped: [batch, frame].
    Returns:
      encoder inputs which can be passed directly to a
      compatible encoder and contains:
      - 'src_inputs': inputs to the encoder, minimally of shape
      [batch, time, ...].
      - 'paddings': a 0/1 tensor of shape [batch, time].
    """
    if paddings is None:
      paddings = tf.zeros_like(audio)

    pcm_audio_data, pcm_audio_paddings = self._reshape_to_mono2d(
      audio, paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._fprop_chunk(
      pcm_audio_data, pcm_audio_paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._pad_and_reshape_spec(
      mel_spectrogram, mel_spectrogram_paddings)

    return mel_spectrogram, mel_spectrogram_paddings

  @staticmethod
  def _stack_signal(signal, stack_size, stride):
    signal = tf.signal.frame(
        signal=signal,
        frame_length=stack_size,
        frame_step=stride,
        pad_end=False,
        axis=1,
    )
    signal = tf.reshape(signal, get_shape(signal)[:2] + [-1])
    return signal

  def _pad_and_reshape_spec(self, mel_spectrogram, mel_spectrogram_paddings):
    p = self.params
    # Stack and sub-sample.
    stack_size = 1
    if p.stack_left_context > 0:
      # Since left context is leading, pad the left by duplicating the first
      # frame.
      stack_size += p.stack_left_context
      mel_spectrogram = tf.concat(
          [mel_spectrogram[:, 0:1, :]] * p.stack_left_context +
          [mel_spectrogram],
          axis=1)
      mel_spectrogram_paddings = tf.concat(
          [mel_spectrogram_paddings[:, 0:1]] * p.stack_left_context +
          [mel_spectrogram_paddings],
          axis=1)

    if p.stack_right_context > 0:
      stack_size += p.stack_right_context
      mel_spectrogram = tf.concat(
          [mel_spectrogram] +
          [mel_spectrogram[:, -1:, :]] * p.stack_right_context,
          axis=1)
      mel_spectrogram_paddings = tf.concat(
          [mel_spectrogram_paddings] +
          [mel_spectrogram_paddings[:, -1:]] * p.stack_right_context,
          axis=1)

    if p.stack_left_context or p.stack_right_context:
      mel_spectrogram = self._stack_signal(mel_spectrogram, stack_size,
                                           p.frame_stride)
      mel_spectrogram_paddings = self._stack_signal(mel_spectrogram_paddings,
                                                    stack_size, p.frame_stride)
      # After stacking paddings, pad if any source frame was padded.
      # Stacks into [batch_size, stacked_frame_dim, stack_size] like the
      # spectrogram stacking above, and then reduces the stack_size dim
      # to the max (effectively, making padding = 1.0 if any of the pre-stacked
      # frames were 1.0). Final shape is [batch_size, stacked_frame_dim].
      mel_spectrogram_paddings = tf.reduce_max(mel_spectrogram_paddings, axis=2)

    # Add feature dim. Shape = [batch, time, features, 1]
    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
    return mel_spectrogram, mel_spectrogram_paddings

  def _apply_preemphasis(self, framed_signal):
    p = self.params
    preemphasized = (
        framed_signal[:, :, 1:] - p.preemph * framed_signal[:, :, 0:-1])
    return preemphasized

  def _get_mel_padding(self, pcm_audio_paddings):
    p = self.params
    # shape: [batch, time, _frame_size]
    framed_paddings = tf.signal.frame(pcm_audio_paddings, self._frame_size,
                                      self._frame_step, p.pad_end)
    # Pad spectrograms that have any padded frames.
    mel_spectrogram_paddings = tf.reduce_max(framed_paddings, axis=2)
    return mel_spectrogram_paddings

  def _fprop_chunk(self, pcm_audio_chunk, pcm_audio_paddings):
    p = self.params
    pcm_audio_chunk = tf.cast(pcm_audio_chunk, tf.float32)
    if p.use_divide_stream:
      pcm_audio_chunk = pcm_audio_chunk / 32768.0

    # shape: [batch, time, _frame_size]
    framed_signal = tf.signal.frame(pcm_audio_chunk, self._frame_size,
                                    self._frame_step, p.pad_end)

    # Pre-emphasis.
    if p.preemph != 0.0:
      preemphasized = self._apply_preemphasis(framed_signal)
    else:
      preemphasized = framed_signal[..., :-1]

    # Noise.
    if p.noise_scale > 0.0:
      noise_signal = tf.random.normal(
          tf.shape(preemphasized),
          stddev=p.noise_scale,
          mean=0.0,
          seed=p.random_seed)
    else:
      noise_signal = 0.0

    # Apply window fn.
    windowed_signal = preemphasized + noise_signal
    if self._window_fn is not None:
      window = self._window_fn(self._frame_size - 1, framed_signal.dtype)
      windowed_signal *= window

    mel_spectrogram = self._mel_spectrogram(windowed_signal)

    mel_spectrogram_log = tf.math.log(
        tf.maximum(float(p.output_floor), mel_spectrogram))

    # Mean and stddev.
    mel_spectrogram_norm = (
        (mel_spectrogram_log - tf.convert_to_tensor(p.per_bin_mean)) /
        tf.convert_to_tensor(p.per_bin_stddev))
    return mel_spectrogram_norm, self._get_mel_padding(pcm_audio_paddings)

  def _mel_spectrogram(self, signal):
    """Computes the mel spectrogram from a waveform signal.
    Args:
      signal: f32 Tensor, shaped [batch_size, num_samples]
    Returns:
      f32 features Tensor, shaped [batch_size, num_frames, mel_channels]
    """
    p = self.params
    # FFT.
    real_frequency_spectrogram = tf.signal.rfft(signal, [self._fft_size])
    magnitude_spectrogram = tf.abs(real_frequency_spectrogram)
    if p.compute_energy:
      magnitude_spectrogram = tf.square(magnitude_spectrogram)

    # Shape of magnitude_spectrogram is num_frames x (fft_size/2+1)
    # Mel_weight is [num_spectrogram_bins, num_mel_bins]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=p.num_bins,
        num_spectrogram_bins=self._fft_size // 2 + 1,
        sample_rate=p.sample_rate,
        lower_edge_hertz=p.lower_edge_hertz,
        upper_edge_hertz=p.upper_edge_hertz,
        dtype=tf.float32)
    # Weight matrix implemented in the magnitude domain.
    batch_size, num_frames, fft_channels = get_shape(magnitude_spectrogram)
    mel_spectrogram = tf.matmul(
        tf.reshape(magnitude_spectrogram,
                   [batch_size * num_frames, fft_channels]), mel_weight_matrix)
    mel_spectrogram = tf.reshape(mel_spectrogram,
                                 [batch_size, num_frames, p.num_bins])

    return mel_spectrogram


class Params:
  """
  Dummy class to emulate Lingvo `Params` as simple as possible.
  """
  def __init__(self):
    self._params = {"name": None}

  def define(self, key, default_value, _comment=None):
    self._params[key] = default_value

  def update(self, d: Dict[str]):
    for key, value in d.items():
      assert key in self._params
      self._params[key] = value

  def items(self):
    return self._params

  def __repr__(self):
    return "Params{%r}" % (self._params,)

  def __setattr__(self, key, value):
    if key == "_params":
      super(Params, self).__setattr__(key, value)
      return
    assert key in self._params
    self._params[key] = value

  def __getattr__(self, item):
    if item == "_params":
      raise AttributeError("__init__ not yet called")
    if item not in self._params:
      raise AttributeError("Params: key %r unknown" % item)
    return self._params[item]

  def assert_same(self, other: Params):
    assert sorted(self._params.keys()) == sorted(other._params.keys()), f"{self} vs {other}"
    for key in self._params.keys():
      assert self._params[key] == other._params[key], f"param {key}: {self._params[key]} vs {other._params[key]}"
