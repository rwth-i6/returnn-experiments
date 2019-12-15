import os
from glob import glob
import soundfile

from sisyphus import *
import sisyphus.toolkit as tk
Path = setup_path(__package__)


class LibriSpeechToBliss(Job):

  def __init__(self, corpus_path, name):
    """
    Generate a bliss xml from  the LibriSpeech corpus.
    :param Path corpus_path:
    :param str name:
    """
    self.corpus_path = corpus_path
    self.name = name
    self.out = self.output_path("out.xml.gz")

  def run(self):

    from recipe.lib.corpus import Corpus, Speaker, Recording, Segment

    corpus = Corpus()
    corpus.name = self.name
    corpus_path = tk.uncached_path(self.corpus_path)
    assert os.path.isdir(corpus_path)

    # the first folder is the speaker id
    for speaker_folder in glob(corpus_path + "/*/"):

      speaker = Speaker()
      speaker.name = os.path.basename(os.path.dirname(speaker_folder))
      print("Add speaker %s" % speaker.name)
      corpus.add_speaker(speaker)

      if os.path.isdir(speaker_folder):
        # the second folder is the audiobook id
        for subfolder in glob(speaker_folder + "/*/"):
          audio_file_dict = {}
          text_dict = {}

          # open the folder to get the audio file names (ids)
          for audio_file in glob(subfolder + "/*.flac"):
            file_id = (audio_file.split("/")[-1]).split(".")[0]
            audio_file_dict[file_id] = audio_file

          # assign the text to the audio file ids
          text_file = glob(subfolder + "/*.trans.txt")[0]
          with open(text_file) as f:
            for line in f:
              file_id, text = line.split(" ", 1)
              text_dict[file_id] = text.strip()

          for file_id in audio_file_dict.keys():
            r = Recording()
            r.name = file_id
            r.audio = audio_file_dict[file_id]
            s = Segment()
            s.name = file_id
            s.speaker_name = speaker.name

            s.orth = text_dict[file_id].strip()

            # open each audio file to get the segment duration
            audio_file = soundfile.SoundFile(audio_file_dict[file_id])
            frames = audio_file._prepare_read(start=0, stop=None, frames=-1)
            audio_duration = frames / audio_file.samplerate
            s.start = 0
            s.end = audio_duration

            r.add_segment(s)
            corpus.add_recording(r)

      print("finished: " + speaker_folder.split("/")[-2])

    corpus.dump(tk.uncached_path(self.out))

  def tasks(self):
    yield Task('run', rqmt={'time': 16})
