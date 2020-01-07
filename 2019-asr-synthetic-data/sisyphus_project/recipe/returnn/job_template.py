from sisyphus import Job, tk

import os
import shutil
import stat
import subprocess as sp

class RETURNNJob(Job):
  """
  Provides common functions for the returnn jobs
  """

  def __init__(self, parameter_dict,
               returnn_config_file,
               returnn_python_exe,
               returnn_root):

    self.returnn_config_file_in = returnn_config_file
    self.returnn_config_file = self.output_path('returnn.config')

    self.parameter_dict = parameter_dict
    if self.parameter_dict is None:
      self.parameter_dict = {}

    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

  def get_parameter_list(self):
    parameter_list = []
    for k, v in sorted(self.parameter_dict.items()):
      if isinstance(v, tk.Variable):
        v = str(v.get())
      elif isinstance(v, tk.Path):
        v = tk.uncached_path(v)
      elif isinstance(v, list):
        v = "\"%s\"" % str(v).replace(" ", "")
      else:
        v = str(v)

      if v.startswith("-"):
        v = "-- " + v

      parameter_list.append("++%s" % k)
      parameter_list.append(v)

    return parameter_list

  def create_files(self):
    # returnn
    shutil.copy(tk.uncached_path(self.returnn_config_file_in),
                tk.uncached_path(self.returnn_config_file))

    parameter_list = self.get_parameter_list()

    with open('rnn.sh', 'wt') as f:
      f.write('#!/usr/bin/env bash\n%s' % ' '.join([tk.uncached_path(self.returnn_python_exe), os.path.join(tk.uncached_path(self.returnn_root), 'rnn.py'), self.returnn_config_file.get_path()] + parameter_list))
    os.chmod('rnn.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  def run(self):
    sp.check_call(["./rnn.sh"])