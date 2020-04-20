from sisyphus import *
import h5py
import numpy
import random

from recipe.lib.hdf import SimpleHDFWriter
from recipe.lib import corpus


class DistributeSpeakerEmbeddings(Job):
  """
  distribute speaker embeddings contained in an hdf file to a new hdf file with mappings to the given bliss corpus
  """

  def __init__(self, bliss_corpus, speaker_embedding_hdf, options=None, use_full_seq_name=False):
    self.bliss_corpus = bliss_corpus
    self.speaker_embedding_hdf = speaker_embedding_hdf
    self.use_full_seq_name = use_full_seq_name
    self.options = options
    if self.options is None:
      self.options = {'mode': 'random'}

    assert self.options['mode'] in ['random', 'length_buckets'], "invalid mode %s" % options['mode']

    self.out = self.output_path("speaker_embeddings.hdf")

  def tasks(self):
    yield Task('run', mini_task=True)

  def _random(self):
    if 'seed' in self.options:
      random.seed(self.options['seed'])

    random.shuffle(self.speaker_embedding_features)
    embedding_index = 0
    for recording in self.c.recordings:
      assert len(recording.segments) == 1
      segment = recording.segments[0] # type:corpus.Segment

      segment_name = "/".join([self.c.name, recording.name, segment.name])

      if not self.use_full_seq_name:
        segment_name = segment.name

      self.hdf_writer.insert_batch(numpy.asarray([self.speaker_embedding_features[embedding_index]]),
                                   [1],
                                   [segment_name])
      embedding_index += 1
      if embedding_index >= len(self.speaker_embedding_features):
        embedding_index = 0

  def _random_matching_length(self):

    text_corpus = corpus.Corpus()
    assert len(text_corpus.subcorpora) == 0
    text_corpus.load(tk.uncached_path(self.options['corpus']))

    text_durations = {}

    max_duration = 0
    for recording in text_corpus.recordings:
      assert len(recording.segments) == 1
      segment = recording.segments[0] # type:corpus.Segment
      segment_name = "/".join([self.c.name, recording.name, segment.name])
      if not self.use_full_seq_name:
        segment_name = segment.name
      seg_len = len(segment.orth)
      text_durations[segment_name] = seg_len
      if seg_len > max_duration:
        max_duration = seg_len

    bucket_size = int(self.options['bucket_size'])
    buckets = [[] for i in range(0, max_duration + bucket_size, bucket_size)]
    bucket_indices = [0] * len(buckets)

    # fill buckets
    for tag, feature in zip(self.speaker_embedding_tags, self.speaker_embedding_features):
      buckets[text_durations[tag] // bucket_size].append(feature)

    # shuffle buckets
    for bucket in buckets:
      random.shuffle(bucket)

    for recording in self.c.recordings:
      assert len(recording.segments) == 1
      segment = recording.segments[0] # type:corpus.Segment
      segment_name = "/".join([self.c.name, recording.name, segment.name])
      if not self.use_full_seq_name:
        segment_name = segment.name

      # search for nearest target bucket
      target_bucket = len(segment.orth) // bucket_size
      for i in range(1000):
        if 0 <= target_bucket + i < len(buckets) and len(buckets[target_bucket + i]) > 0:
          target_bucket = target_bucket + i
          break
        if 0 <= target_bucket - i < len(buckets) and len(buckets[target_bucket - i]) > 0:
          target_bucket = target_bucket - i
          break

      speaker_embedding = buckets[target_bucket][bucket_indices[target_bucket]]
      self.hdf_writer.insert_batch(numpy.asarray([speaker_embedding]),
                                   [1],
                                   [segment_name])
      bucket_indices[target_bucket] += 1
      if bucket_indices[target_bucket] >= len(buckets[target_bucket]):
        bucket_indices[target_bucket] = 0

  def run(self):

    speaker_embedding_data = h5py.File(tk.uncached_path(self.speaker_embedding_hdf), 'r')
    speaker_embedding_inputs = speaker_embedding_data['inputs']
    speaker_embedding_raw_tags = speaker_embedding_data['seqTags']
    speaker_embedding_lengths = speaker_embedding_data['seqLengths']

    self.speaker_embedding_features = []
    self.speaker_embedding_tags = []
    offset = 0
    for tag, length in zip(speaker_embedding_raw_tags, speaker_embedding_lengths):
      self.speaker_embedding_features.append(speaker_embedding_inputs[offset:offset+length[0]])
      self.speaker_embedding_tags.append(tag.decode() if isinstance(tag, bytes) else tag)
      offset += length[0]

    self.hdf_writer = SimpleHDFWriter(tk.uncached_path(self.out), dim=self.speaker_embedding_features[0].shape[-1])

    self.c = corpus.Corpus()
    self.c.load(tk.uncached_path(self.bliss_corpus))
    assert len(self.c.subcorpora) == 0

    mode = self.options.get('mode')
    if mode == "random":
      self._random()
    elif mode == "length_buckets":
      self._random_matching_length()
    else:
      assert False

    self.hdf_writer.close()


class VerifyCorpus(Job):
  """
  verifies the audio files of a bliss corpus by loading it with the soundfile library
  """

  def __init__(self, bliss_corpus, channels=1, sample_rate=16000):
    self.bliss_corpus = bliss_corpus
    self.channels = channels
    self.sample_rate = sample_rate

    self.out = self.output_path("errors.log")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    import soundfile

    c = corpus.Corpus()
    c.load(tk.uncached_path(self.bliss_corpus))

    out_file = open(tk.uncached_path(self.out), "wt")

    success = True

    for r in c.all_recordings():
      try:
        audio, sr = soundfile.read(open(r.audio, "rb"))
        if self.channels == 1:
          assert len(audio.shape) == 1
        else:
          assert audio.shape[1] == self.channels
        assert sr == self.sample_rate
      except Exception as e:
        print("error in file %s: %s" % (r.audio, str(e)))
        out_file.write("error in file %s: %s\n" % (r.audio, str(e)))
        success = False

    assert success, "there was an error, please see error.log"