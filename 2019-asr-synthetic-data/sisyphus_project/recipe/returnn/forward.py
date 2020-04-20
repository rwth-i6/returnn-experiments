__all__ = ['RETURNNForwardFromFile']

from sisyphus import *
Path = setup_path(__package__)

import os
import stat
import subprocess as sp

from recipe.default_values import RETURNN_PYTHON_EXE, RETURNN_SRC_ROOT
from recipe.returnn.job_template import RETURNNJob


class RETURNNForwardFromFile(RETURNNJob):
  def __init__(self, returnn_config_file, parameter_dict, hdf_outputs,
               time_rqmt=4, mem_rqmt=4,
               returnn_python_exe=RETURNN_PYTHON_EXE, returnn_root=RETURNN_SRC_ROOT):

    super().__init__(parameter_dict,
                     returnn_config_file,
                     returnn_python_exe,
                     returnn_root)

    self.rqmt = { 'gpu' : 1, 'cpu' : 2, 'mem' : mem_rqmt, 'time' : time_rqmt }

    self.parameter_dict['forward_override_hdf_output'] = True

    self.outputs = {}
    for output in hdf_outputs:
      self.outputs[output] = self.output_path(output + ".hdf")

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

  def run(self):
    super().run()
    for k,v in self.outputs.items():
      self.sh("mv %s %s" % (k + ".hdf", v))

    # delete dumped file and hdf files that were not marked as output, if remaining
    self.sh("rm dump*", pipefail=False, except_return_codes=(0,1))
    self.sh("rm *.hdf", pipefail=False, except_return_codes=(0,1))

  @classmethod
  def hash(cls, kwargs):

    d = { 'returnn_config_file'     : kwargs['returnn_config_file'],
          'parameter_dict'     : kwargs['parameter_dict'],
          'returnn_python_exe' : kwargs['returnn_python_exe'],
          'returnn_root'       : kwargs['returnn_root'],}

    return super().hash(d)