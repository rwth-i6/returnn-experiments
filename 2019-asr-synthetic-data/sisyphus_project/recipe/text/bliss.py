from sisyphus import *

import gzip

from recipe.lib.corpus import Corpus

class BlissExtractRawText(Job):
  """
  Extract the Text from a Bliss corpus into a raw gziptext file
  """
  def __init__(self, corpus, segments=None, segment_key_only=True):
    self.corpus_path = corpus
    self.out = self.output_path("text.gz")
    self.segments_file_path = segments

    self.segment_key_only = segment_key_only

  def tasks(self):
    yield Task('run', mini_task=True)


  def run(self):
    import gzip
    corpus = Corpus()
    corpus.load(tk.uncached_path(self.corpus_path))

    outfile = gzip.open(tk.uncached_path(self.out), "wt")

    segments = None
    if self.segments_file_path:
      if tk.uncached_path(self.segments_file_path).endswith("gz"):
        segment_file = gzip.open(tk.uncached_path(self.segments_file_path), "rb")
      else:
        segment_file = open(tk.uncached_path(self.segments_file_path), "rt")
      segments = [line.decode().strip() for line in segment_file]

    for recording in corpus.all_recordings():
      for segment in recording.segments:
        full_segment_key = "/".join([corpus.name, recording.name, segment.name])
        if segments:
          if full_segment_key not in segments:
            continue

        orth = segment.orth.strip() + "\n"
        outfile.write(orth)

    outfile.close()