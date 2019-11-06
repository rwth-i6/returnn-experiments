from sisyphus import *

import gzip
import os

from recipe.lib import corpus


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