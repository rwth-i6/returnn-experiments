from sisyphus import *

import gzip
import os

from recipe.lib import corpus
from recipe.util import chunks


class BlissToZipDataset(Job):
  """
  convert bliss corpus with single segment recordings into the Zip format for RETURNN training
  """

  def __init__(self, name, corpus_file, segment_file=None, use_full_seq_name=False, no_audio=False):
    """

    :param str name:
    :param str|Path corpus_file:
    :param str|Path segment_file:
    :param bool use_full_seq_name: only use the segment name as sequence name, compatible to old use file as name behavior
    :param bool no_audio:
    """
    self.name = name
    self.corpus_file = corpus_file
    self.segment_file_path = segment_file
    self.use_full_seq_name = use_full_seq_name
    self.no_audio = no_audio
    self.out = self.output_path("%s.zip" % name)

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    import zipfile
    zip_file = zipfile.ZipFile(tk.uncached_path(self.out), mode='w', compression=zipfile.ZIP_STORED)
    dict_file_path = self.name + ".txt"
    dict_file = open(dict_file_path, "wt")
    dict_file.write("[\n")
    c = corpus.Corpus()
    assert len(c.subcorpora) == 0
    c.load(tk.uncached_path(self.corpus_file))

    if self.segment_file_path:
      if tk.uncached_path(self.segment_file_path).endswith("gz"):
        segment_file = gzip.open(tk.uncached_path(self.segment_file_path), "rb")
      else:
        segment_file = open(tk.uncached_path(self.segment_file), "rt")
      segments = [line.decode().strip() for line in segment_file]

    for recording in c.recordings:
      # skip empty recordings
      if not recording.segments:
        continue

      # validate that each recording only contains one segment
      assert len(recording.segments) == 1
      segment = recording.segments[0] # type:corpus.Segment

      segment_name = "/".join([c.name, recording.name, segment.name])
      if self.segment_file_path and segment_name not in segments:
        continue

      if not self.use_full_seq_name:
        segment_name = segment.name

      if self.no_audio:
        dict_file.write('{"duration": %f, "text": "%s", "seq_name": "%s"},\n'
                        % (segment.end,
                           segment.orth.replace('"', '\\"'),
                           segment_name))
      else:
        audio_path = recording.audio
        arc_path = os.path.join(self.name, os.path.basename(audio_path))
        zip_file.write(audio_path, arcname=arc_path)
        dict_file.write('{"file": "%s", "duration": %f, "text": "%s", "seq_name": "%s"},\n'
                        % (os.path.basename(audio_path),
                           segment.end,
                           segment.orth.replace('"', '\\"'),
                           segment_name))

    dict_file.write(']\n')
    dict_file.close()

    zip_file.write(dict_file_path, dict_file_path)
    zip_file.close()


class MergeCorpora(Job):
  """
  Merges Bliss Corpora into a single file as subcorpora
  This is preferably done after using corpus compression

  :param Iterable[Path] corpora: any iterable of bliss corpora file paths to merge
  :param name: name of the new corpus (subcorpora will keep the original names)
  """
  def __init__(self, corpora, name, subcorpora=True):

    self.corpora = corpora
    self.name = name
    self.subcorpora = subcorpora
    self.merged_corpus = self.output_path("merged.xml.gz")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    merged_corpus = corpus.Corpus()
    merged_corpus.name = self.name
    for corpus_path in self.corpora:
      c = corpus.Corpus()
      c.load(str(corpus_path))
      if self.subcorpora:
        merged_corpus.add_subcorpus(c)
      else:
        for rec in c.all_recordings():
          merged_corpus.add_recording(rec)

    merged_corpus.dump(tk.uncached_path(self.merged_corpus))


class SegmentCorpus(Job):
  def __init__(self, corpus_path, num_segments, use_fullname=False):
    self.set_vis_name('Segment Corpus')

    self.corpus_path = corpus_path
    self.num_segments = num_segments
    self.use_fullname = use_fullname
    self.segment_files = [self.output_path('segments.%d' % i) for i in range(num_segments)]

  def tasks(self):
    yield Task('run', resume='run', mini_task=True)

  def run(self):
    c = corpus.Corpus()
    c.load(tk.uncached_path(self.corpus_path))

    all_segments = list(c.segments())

    for idx, segments in enumerate(chunks(all_segments, self.num_segments)):
      with open(self.segment_files[idx].get_path(), 'wt') as segment_file:
        for segment in segments:
          if self.use_fullname:
            segment_file.write(segment.fullname() + '\n')
          else:
            segment_file.write(segment.name + '\n')


class BlissAddTextFromBliss(Job):

  """
  This Job is used to add the text to a bliss corpus containing only audio from another bliss corpus
  containing the same sequences.
  """

  def __init__(self, empty_bliss_corpus, bliss_corpus):
    self.empty_bliss_corpus = empty_bliss_corpus
    self.bliss_corpus = bliss_corpus

    self.out = self.output_path("corpus.xml.gz")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    orth_c = corpus.Corpus()
    orth_c.load(tk.uncached_path(self.bliss_corpus))

    orths = {}
    for r in orth_c.all_recordings():
      assert len(r.segments) == 1, "needs to be a single segment recording"
      orth = r.segments[0].orth
      tag = r.segments[0].name
      orths[tag] = orth

    c = corpus.Corpus()
    c.load(tk.uncached_path(self.empty_bliss_corpus))

    for r in c.all_recordings():
      assert len(r.segments) == 1, "needs to be a single segment recording"
      tag = r.segments[0].name
      orth = orths[tag]
      r.segments[0].orth = orth

    c.dump(tk.uncached_path(self.out))