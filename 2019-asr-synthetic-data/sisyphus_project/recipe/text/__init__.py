import subprocess as sp
from sisyphus import *

class Concatenate(Job):

  """ Concatenate all given input files """

  def __init__(self, inputs):
    assert(inputs)

    # ensure sets are always merged in the same order
    if isinstance(inputs, set):
      inputs = list(inputs)
      inputs.sort(key=lambda x: str(x))

    assert(isinstance(inputs, list))

    # Skip this job if only one input is present
    if len(inputs) == 1:
      self.out = inputs.pop()
    else:
      self.out = self.output_path('out.gz')

    for input in inputs:
      assert isinstance(input, Path) or isinstance(input, str), "input to Concatenate is not a valid path"

    self.inputs = inputs

  def run(self):
    self.f_list = ' '.join(str(i) for i in self.inputs)
    self.sh('zcat -f {f_list} | gzip > {out}')

  def tasks(self):
    yield Task('run', rqmt={'mem': 3, 'time': 3})