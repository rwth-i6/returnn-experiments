import os

from sisyphus import Job, Task, tk
from recipe.lib import corpus


class BlissFFMPEGJob(Job):
  """
  Changes the speed of all audio files in the corpus (shifting time AND frequency)

  """

  def __init__(self, corpus_file, ffmpeg_option_string, ffmpeg_binary=None, output_format=None):
    self.corpus_file = corpus_file
    self.ffmpeg_option_string = ffmpeg_option_string
    self.audio_folder = self.output_path("audio/", directory=True)
    self.out = self.output_path("corpus.xml.gz")
    self.rqmt = {'time': 4, 'cpu': 4, 'mem': 8}
    self.ffmpeg_binary = ffmpeg_binary if ffmpeg_binary else "ffmpeg"
    self.output_format = output_format

  def tasks(self):
    yield Task('run', rqmt=self.rqmt)

  def perform_ffmpeg(self, r):
    audio_name = r.audio.split("/")[-1]

    if self.output_format is not None:
      name, ext = os.path.splitext(audio_name)
      audio_name = name + "." + self.output_format

    target = tk.uncached_path(self.audio_folder) + "/" + audio_name
    seconds = None
    if not os.path.exists(target):
      result = self.sh(
        "%s -hide_banner -y -i %s %s {audio_folder}/%s" % (self.ffmpeg_binary,
                                                           r.audio, self.ffmpeg_option_string, audio_name), include_stderr=True)
    else:
      print("found %s" % target)
    return seconds

  def run(self):
    c = corpus.Corpus()
    nc = corpus.Corpus()

    c.load(tk.uncached_path(self.corpus_file))
    nc.name = c.name
    nc.speakers = c.speakers
    nc.default_speaker = c.default_speaker
    nc.speaker_name = c.speaker_name
    # store index of last segment
    for r in c.recordings:
      nr = corpus.Recording()
      nr.name = r.name
      nr.segments = r.segments
      nr.speaker_name = r.speaker_name
      nr.speakers = r.speakers
      nr.default_speaker = r.default_speaker

      audio_name = r.audio.split("/")[-1]

      if self.output_format is not None:
        name, ext = os.path.splitext(audio_name)
        audio_name = name + "." + self.output_format

      nr.audio = os.path.join(tk.uncached_path(self.audio_folder), audio_name)
      nc.add_recording(nr)


    from multiprocessing import pool
    p = pool.Pool(4)
    p.map(self.perform_ffmpeg, c.recordings)

    nc.dump(tk.uncached_path(self.out))