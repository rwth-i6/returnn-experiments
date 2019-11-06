__all__ = ['RETURNNSearchFromFile', 'ReturnnScore', 'SearchBPEtoWords', 'SearchWordsToCTM']

from sisyphus import *
Path = setup_path(__package__)

import os
import stat
import subprocess as sp

from recipe.default_values import RETURNN_PYTHON_EXE, RETURNN_SRC_ROOT



class RETURNNSearchFromFile(Job):
  def __init__(self, returnn_config_file, parameter_dict, output_mode="py",
               time_rqmt=4, mem_rqmt=4,
               returnn_python_exe=None, returnn_root=None):

    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.returnn_config_file_in = returnn_config_file
    self.parameter_dict = parameter_dict
    if self.parameter_dict is None:
      self.parameter_dict = {}

    self.returnn_config_file = self.output_path('returnn.config')

    self.rqmt = { 'gpu' : 1, 'cpu' : 2, 'mem' : mem_rqmt, 'time' : time_rqmt }

    assert output_mode in ['py', 'txt']
    self.out = self.output_path("search_out.%s" % output_mode)

    self.parameter_dict['search_output_file'] = tk.uncached_path(self.out)
    self.parameter_dict['search_output_file_format'] = output_mode

  def update(self):
    if "ext_model" in self.parameter_dict.keys() and "ext_load_epoch" in self.parameter_dict.keys():
      epoch = self.parameter_dict['ext_load_epoch']
      epoch = epoch.get() if isinstance(epoch, tk.Variable) else epoch
      model_dir = self.parameter_dict['ext_model']
      if isinstance(model_dir, tk.Path):
        self.add_input(Path(str(model_dir) + "/epoch.%03d.index" % epoch, creator=model_dir.creator))
      else:
        self.add_input(Path(str(model_dir) + "/epoch.%03d.index" % epoch))

  def tasks(self):
    yield Task('create_files', mini_task=True)
    yield Task('run', resume='run', rqmt=self.rqmt)

  def get_parameter_list(self):
    parameter_list = []
    for k,v in sorted(self.parameter_dict.items()):
      if isinstance(v, tk.Variable):
        v = str(v.get())
      elif isinstance(v, tk.Path):
        v = tk.uncached_path(v)
      else:
        v = str(v)
      if k == "ext_model" and not v.endswith("/epoch"):
        v = v + "/epoch"
      parameter_list.append("++%s" % k)
      parameter_list.append(v)

    return parameter_list

  def create_files(self):
    # returnn
    self.sh("cp {returnn_config_file_in} {returnn_config_file}")

    parameter_list = self.get_parameter_list()

    with open('rnn.sh', 'wt') as f:
      f.write('#!/usr/bin/env bash\n%s' % ' '.join([tk.uncached_path(self.returnn_python_exe), os.path.join(tk.uncached_path(self.returnn_root), 'rnn.py'), self.returnn_config_file.get_path()] + parameter_list))
    os.chmod('rnn.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  def run(self):
    parameter_list = self.get_parameter_list()

    sp.check_call([tk.uncached_path(self.returnn_python_exe), os.path.join(tk.uncached_path(self.returnn_root), 'rnn.py'), self.returnn_config_file.get_path()] + parameter_list)

  @classmethod
  def hash(cls, kwargs):

    d = { 'returnn_config_file'     : kwargs['returnn_config_file'],
          'parameter_dict'     : kwargs['parameter_dict'],
          'returnn_python_exe' : kwargs['returnn_python_exe'],
          'returnn_root'       : kwargs['returnn_root'],
          'output_mode': kwargs['output_mode']}

    return super().hash(d)


class GetBestEpoch(Job):

  __sis_hash_exclude__ = {'key': None}

  def __init__(self, model_dir, learning_rates, index=0, key=None):
    self.model_dir = model_dir
    self.learning_rates = learning_rates
    self.index = index
    self.out_var = self.output_var("epoch")
    self.key = key
    assert index >= 0 and isinstance(index, int)

  def run(self):
    def EpochData(learningRate, error):
      return {'learning_rate': learningRate, 'error': error}

    with open(self.learning_rates.get_path(), 'rt') as f:
      text = f.read()

    data = eval(text)

    epochs = list(sorted(data.keys()))

    if self.key == None:
      dev_score_keys = [k for k in data[epochs[-1]]['error'] if k.startswith('dev_score')]
      dsk = dev_score_keys[0]
    else:
      dsk = self.key

    dev_scores = [(epoch, data[epoch]['error'][dsk]) for epoch in epochs if dsk in data[epoch]['error']]

    sorted_scores = list(sorted(dev_scores, key=lambda x: x[1]))

    print(sorted_scores)

    self.out_var.set(sorted_scores[self.index][0])

  def tasks(self):
    yield Task('run', mini_task=True)


class SearchBPEtoWords(Job):
  """
  Converts BPE Search output from returnn into words
  :param search_output:
  :param script:
  """
  def __init__(self, search_output_bpe, script=Path("scripts/search-bpe-to-words.py")):

    self.search_output_bpe = search_output_bpe
    self.script = script
    self.out = self.output_path("search_output.words")

  def run(self):
    self.sh("python3 {script} {search_output_bpe} --out {out}")

  def tasks(self):
    yield Task('run', mini_task=True)


class SearchWordsToCTM(Job):
  """
  Converts search output (in words) from returnn into a ctm file
  :param search_output:
  :param script:
  """

  __sis_hash_exclude__ = {"only_segment_name": False}

  def __init__(self, search_output_words, corpus, only_segment_name=False, script=Path("scripts/search-words-to-ctm.py")):
    self.search_output_words = search_output_words
    self.corpus = corpus
    self.script = script
    self.only_segment_name = only_segment_name
    self.out = self.output_path("search_output.ctm")

  def run(self):
    self.sh("python3 {script} {search_output_words} --corpus {corpus} %s --out {out}" % ("--only-segment-name" if self.only_segment_name else ""))

  def tasks(self):
    yield Task('run', mini_task=True)


class ReturnnScore(Job):

  def __init__(self, hypothesis, reference, returnn_python_exe=RETURNN_PYTHON_EXE, returnn_root=RETURNN_SRC_ROOT):
    self.set_attrs(locals())
    self.out = self.output_path("wer")

  def run(self):
    call = [str(self.returnn_python_exe), os.path.join(str(self.returnn_root), 'tools/calculate-word-error-rate.py'),
            "--expect_full",
            "--hyps",
            str(self.hypothesis), "--refs", str(self.reference), "--out", str(self.out)]
    print("run %s" % " ".join(call))
    sp.check_call(call)

  def tasks(self):
    yield Task('run', mini_task=True)
