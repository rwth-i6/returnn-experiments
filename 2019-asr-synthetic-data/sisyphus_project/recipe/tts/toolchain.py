from sisyphus import *

import h5py
import numpy
import os

from multiprocessing import pool
from recipe.lib.hdf import SimpleHDFWriter
from recipe.tts.lib.griffin_lim import GLConverter

class ConvertFeatures(Job):
  """
  Convert output features of a decoding job that have merged frames to single frames
  """

  def __init__(self, stacked_hdf_features, conversion_factor):
    """

    :param Path stacked_hdf_features: hdf features with stacked frames
    :param int conversion_factor: the number of frames per stack
    """
    self.stacked_hdf_features = stacked_hdf_features
    self.conversion_factor = conversion_factor

    self.out = self.output_path("features.hdf")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    input_data = h5py.File(tk.uncached_path(self.stacked_hdf_features), 'r')
    inputs = input_data['inputs']
    tags = input_data['seqTags']
    lengths = input_data['seqLengths']

    print("loaded arrays")
    dim = inputs[0].shape[0] // self.conversion_factor
    print("out dim: %d" % dim)

    writer = SimpleHDFWriter(tk.uncached_path(self.out), dim=dim, ndim=2)

    offset = 0
    for tag, length in zip(tags, lengths):
      in_data = inputs[offset:offset+length[0]]
      dim = int(in_data.shape[1] / int(self.conversion_factor))
      out_data = numpy.reshape(in_data, (-1, dim), order='C')
      offset += length[0]
      writer.insert_batch(numpy.asarray([out_data]), [out_data.shape[0]], [tag])

    writer.close()


class GriffinLim(Job):
  """
  Run Griffin & Lim algorithm on linear spectogram features with specified audio settings
  """

  def __init__(self,
               linear_features,
               iterations,
               sample_rate,
               window_shift,
               window_size,
               preemphasis,
               file_format="ogg",
               corpus_format="bliss"):
    """

    :param linear_features:
    :param iterations:
    :param sample_rate:
    :param window_shift:
    :param window_size:
    :param preemphasis:
    :param file_format:
    :param corpus_format:
    :param input_hdf:
    """
    self.set_attrs(locals())

    self.out_folder = self.output_path("audio", directory=True)

    if corpus_format == "bliss":
      self.out_corpus = self.output_path("audio/corpus.xml.gz")
    else:
      self.out_corpus = None

    self.rqmt = {'cpu': 4, 'mem': 8, 'time': 12}

  def tasks(self):
    yield Task('run', rqmt=self.rqmt)

  def run(self):
    assert os.path.isdir(tk.uncached_path(self.out_folder))
    assert self.file_format in ["wav", "ogg"]
    assert self.corpus_format in ["bliss", "json", None]

    ref_linear_data = h5py.File(tk.uncached_path(self.linear_features), 'r')
    rl_inputs = ref_linear_data['inputs']
    rl_tags = ref_linear_data['seqTags']
    rl_lengths = ref_linear_data['seqLengths']

    n_fft = rl_inputs[0].shape[0]*2
    print("N_FFT from HDF: % i" % n_fft)

    converter = GLConverter(out_folder=tk.uncached_path(self.out_folder),
                            out_corpus=tk.uncached_path(self.out_corpus),
                            sample_rate=self.sample_rate,
                            window_shift=self.window_shift,
                            window_size=self.window_size,
                            n_fft=n_fft,
                            iterations=self.iterations,
                            preemphasis=self.preemphasis,
                            file_format=self.file_format,
                            corpus_format=self.corpus_format)

    # H5py has issues with multithreaded loading, so buffer 512 spectograms
    # single threaded and then distribute to the workers for conversion

    p = pool.Pool(4)

    loaded_spectograms = []
    offset = 0
    group = 0
    for i, (tag, length) in enumerate(zip(rl_tags, rl_lengths)):
      tag = tag if isinstance(tag, str) else tag.decode()
      loaded_spectograms.append((tag, numpy.asarray(rl_inputs[offset:offset + length[0]])))
      offset += length[0]
      if len(loaded_spectograms) > 512:
        print("decode group %i" % group)
        group += 1
        recordings = p.map(converter.convert, loaded_spectograms)

        # put all recordings to the corpus
        if self.corpus_format == "bliss":
          for recording in recordings:
            converter.corpus.add_recording(recording)

        # force gc for minimal memory requirement
        del loaded_spectograms
        loaded_spectograms = []

    # process rest in the buffer
    if len(loaded_spectograms) > 0:
      recordings = p.map(converter.convert, loaded_spectograms)

      # put all recordings to the corpus
      if self.corpus_format == "bliss":
        for recording in recordings:
          converter.corpus.add_recording(recording)


    converter.save_corpus()