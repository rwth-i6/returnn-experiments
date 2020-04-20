#!python3
import os
import errno
import subprocess
from recipe.lib import corpus as bliss_corpus
import tempfile
from sisyphus import gs

try:
  import numpy as np
  import librosa
  from scipy import signal
  from scipy.io import wavfile
  import soundfile as sf
except:
  assert False, "Griffin & Lim conversion requires librosa, numpy, scipy and soundfile "

class GLConverter():

  def __init__(self, out_folder, out_corpus, sample_rate, window_shift, window_size, n_fft, iterations, preemphasis, file_format, corpus_format):
    self.out_folder = out_folder
    self.sample_rate = sample_rate
    self.window_shift = window_shift
    self.window_size = window_size
    self.n_fft = n_fft
    self.iterations = iterations
    self.preemphasis = preemphasis
    self.file_format = file_format
    self.corpus_format = corpus_format

    if self.corpus_format == "bliss":
      self.corpus_path = out_corpus
      self.corpus = bliss_corpus.Corpus()
      self.corpus.name = "GRIFFIN_LIM"

    self.tmp_path = tempfile.mkdtemp(prefix=gs.TMP_PREFIX)

  def _istft(self, y):
    return librosa.istft(y, hop_length=int(self.sample_rate*self.window_shift), win_length=int(self.sample_rate*self.window_size))

  def _stft(self, y):
    return librosa.stft(y=y, n_fft=self.n_fft, hop_length=int(self.sample_rate*self.window_shift), win_length=int(self.sample_rate*self.window_size), pad_mode='constant')

  def griffin_lim(self, S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = self._istft(S_complex * angles)
    for i in range(self.iterations):
      angles = np.exp(1j * np.angle(self._stft(y)))
      y = self._istft(S_complex * angles)
    return y

  def convert(self, index_tuple):
    tag = index_tuple[0]
    lin_spec = index_tuple[1]
    if "/" in tag:
      tag = tag.split("/")[-1]

    lin_spec = lin_spec.T
    lin_spec_pp = np.pad(lin_spec, ((1,0),(0,0)), mode='constant', constant_values=0)
    lin_spec_pp[0, :] = 0
    lin_spec_pp[1, :] = 0
    inv_spec = inv_preemphasis(self.griffin_lim(lin_spec_pp), self.preemphasis, self.preemphasis != 0)
    if self.file_format == "wav":
      path = os.path.join(self.out_folder, "%s.wav" % tag)
      save_wav(inv_spec, path, self.sample_rate)
      assert os.path.exists(path)
    elif self.file_format == "ogg":
      # export as wav to shared memory
      tmp_path = os.path.join(self.tmp_path, "%s.wav" % tag)
      save_wav(inv_spec, tmp_path, self.sample_rate)
      path = os.path.join(self.out_folder, '%s.ogg' % tag)

      assert os.path.exists(tmp_path)
      # use ffmpeg to convert to ogg
      try:
        output = subprocess.check_output(['ffmpeg', '-hide_banner', '-i', tmp_path, path], stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
      else:
        print(output.decode())
      subprocess.run(['rm', tmp_path], check=True)
      assert os.path.exists(path)
    else:
      assert False

    if self.corpus_format == "bliss":
      recording = bliss_corpus.Recording()
      recording.name = tag
      recording.audio = path
      segment = bliss_corpus.Segment()
      segment.name = tag
      segment.start = 0
      segment.end = float(len(inv_spec)) / float(self.sample_rate)
      recording.add_segment(segment)
      return recording
    elif self.corpus_format == "json":
      return NotImplementedError

    return None

  def save_corpus(self):
    if self.corpus_format == "bliss":
      self.corpus.dump(self.corpus_path)


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def inv_preemphasis(wav, k, inv_preemphasize=True):
  if inv_preemphasize:
    return signal.lfilter([1], [1, -k], wav)
  return wav


def save_wav(wav, path, sr):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  #proposed by @dsmiller
  wavfile.write(path, sr, wav.astype(np.int16))
