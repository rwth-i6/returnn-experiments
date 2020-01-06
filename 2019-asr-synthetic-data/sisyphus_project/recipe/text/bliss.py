from sisyphus import *

import gzip

from recipe.lib.corpus import Corpus
import gzip
import re
import pickle
from sisyphus import *
from recipe.lib.corpus import *
import string


class PP_Module(object):

  def __init__(self, **kwargs):
    pass

  def process(self, orth: str):
    return orth


class Lowercase(PP_Module):
  def process(self, orth: str):
    return orth.lower()

class Uppercase(PP_Module):
  def process(self, orth: str):
    return orth.upper()

class End_Token(PP_Module):
  def __init__(self, token):
    super().__init__()
    self.token = token
    assert len(token) == 1 and isinstance(token, str)

  def process(self, orth: str):
    return orth + self.token

class RemoveSymbol(PP_Module):
  def __init__(self, symbol):
    super().__init__()
    self.symbol = symbol
    assert len(symbol) == 1

  def process(self, orth: str):
    return orth.replace(self.symbol, "")

class RemovePunctuation(PP_Module):

  def __init__(self):
    super().__init__()
    self.converter = str.maketrans('', '', string.punctuation)

  def process(self, orth: str):
    return orth.translate(self.converter)

class RegexReplace(PP_Module):
  def __init__(self, search, replace):
    super().__init__()
    self.regex_search = search
    self.regex_replace = replace

  def process(self, orth: str):
    return re.sub(self.regex_search, self.regex_replace, orth)


module_dict = {'lowercase': Lowercase,
               'uppercase': Uppercase,
               'end_token': End_Token,
               'remove_symbol': RemoveSymbol,
               'regex_replace': RegexReplace,
               'remove_punctuation': RemovePunctuation}


class ProcessBlissText(Job):
  """
  Provides a set of processing modules to process the orth tags in a bliss corpus file
  """

  def __init__(self, corpus, process_list, vocabulary=None):
    """

    :param Path corpus: path to the corpus file
    :param list[(str, dict)] process_list: list of module tuples with (module_name, parameter_dict)
    :param Path vocabulary: a character vocabulary to be used as whitelist
    """
    self.corpus = corpus
    self.process_list = process_list
    self.out = self.output_path("pp_corpus.xml.gz")
    self.vocabulary = vocabulary

    self.module_instances = []
    for (module, params) in process_list:
      assert module in module_dict.keys(), "module %s was not in the list of available modules %s" % (module, sorted(module_dict.keys()))
      module_instance = module_dict[module](**params)
      self.module_instances.append(module_instance)

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    corpus = Corpus()
    corpus.load(tk.uncached_path(self.corpus))

    if self.vocabulary:
      vocab = pickle.load(open(tk.uncached_path(self.vocabulary), "rb"))

    for recording in corpus.all_recordings():
      for segment in recording.segments:
        orth = segment.orth.strip()
        for module_instance in self.module_instances:
          orth = module_instance.process(orth)
        if self.vocabulary:
          orth = "".join([c for c in orth if c in vocab.keys()])
        segment.orth = orth

    corpus.dump(tk.uncached_path(self.out))


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


class BlissExtractTextDictionary(Job):
  """
  Extract the Text from a Bliss corpus to fit the "{key: text}" structure
  """

  def __init__(self, corpus, segments=None, segment_key_only=True):
    """

    :param corpus: bliss corpus file
    :param segments: a segment file as optional whitelist
    :param segment_key_only: if true, only the segment is used as key, instead of corpus/recording/segment
    """
    self.corpus_path = corpus
    self.out = self.output_path("text.py")
    self.segments_file_path = segments

    self.segment_key_only = segment_key_only

  def tasks(self):
    yield Task('run', mini_task=True)


  def run(self):
    import pprint
    corpus = Corpus()
    corpus.load(tk.uncached_path(self.corpus_path))

    dictionary = {}

    segments = None
    if self.segments_file_path:
      if tk.uncached_path(self.segments_file_path).endswith("gz"):
        segment_file = gzip.open(tk.uncached_path(self.segments_file_path), "rb")
      else:
        segment_file = open(tk.uncached_path(self.segments_file_path), "rt")
      segments = [line.decode().strip() for line in segment_file]

    for recording in corpus.all_recordings():
      for segment in recording.segments:
        orth = segment.orth.strip()
        full_segment_key = "/".join([corpus.name, recording.name, segment.name])
        key = segment.name if self.segment_key_only else full_segment_key

        if segments:
          if full_segment_key not in segments:
            continue

        dictionary[key] = orth

    dictionary_string = pprint.pformat(dictionary, width=1000)

    outfile = open(tk.uncached_path(self.out), "wt")
    outfile.write(dictionary_string)
    outfile.close()
