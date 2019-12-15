__all__ = ['ExtractDatasetStats']

from sisyphus import *
Path = setup_path(__package__)

import os
import stat
import subprocess
import numpy

from recipe.default_values import RETURNN_PYTHON_EXE, RETURNN_SRC_ROOT

from recipe.returnn.config import RETURNNConfig

class ExtractDatasetStats(Job):

  def __init__(self, config, returnn_python_exe=RETURNN_PYTHON_EXE, returnn_root=RETURNN_SRC_ROOT):

    self.config = RETURNNConfig(config, {})
    self.crnn_python_exe = returnn_python_exe
    self.crnn_root = returnn_root

    self.mean = self.output_var("mean_var")
    self.std_dev = self.output_var("std_dev_var")

    self.mean_file = self.output_path("mean")
    self.std_dev_file = self.output_path("std_dev")

  def tasks(self):
    yield Task('run', rqmt={'cpu':1, 'mem': 4, 'time': 4}, mini_task=True)

  def run(self):
    self.config.write("crnn.config")

    with open('rnn.sh', 'wt') as f:
      f.write('#!/usr/bin/env bash\n%s' % ' '.join([tk.uncached_path(self.crnn_python_exe),
                                                    os.path.join(tk.uncached_path(self.crnn_root), 'tools/dump-dataset.py'),
                                                    "crnn.config",
                                                    "--endseq -1",
                                                    "--stats",
                                                    "--dump_stats stats"]))
    os.chmod('rnn.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # there can be a weird error when python thinks the terminal does not accept UTF-8
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "UTF-8"
    subprocess.check_call(["./rnn.sh"], env=env)

    self.sh("cp stats.mean.txt {mean_file}")
    self.sh("cp stats.std_dev.txt {std_dev_file}")

    total_mean = 0
    total_var = 0

    mean_file = open("stats.mean.txt")
    std_dev_file = open("stats.std_dev.txt")

    for i,(mean, std_dev) in enumerate(zip(mean_file, std_dev_file)):

      mean = float(mean)
      var = float(std_dev.strip())**2
      print(var)
      total_mean = (total_mean*i + mean) / (i+1)
      total_var = (total_var*i +  var + (total_mean - mean)**2 * i / (i+1)) / (i + 1)
      print(total_var)


    self.mean.set(total_mean)
    self.std_dev.set(numpy.sqrt(total_var))

    print(numpy.sqrt(total_var))

