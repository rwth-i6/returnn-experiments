__all__ = ['RETURNNConfig', 'WriteRETURNNConfigJob']

from sisyphus import *
Path = setup_path(__package__)
Variable = tk.Variable

import json
import pprint
import string
import textwrap


def instanciate_vars(o):
  if isinstance(o, Variable):
    o = o.get()
  elif isinstance(o, list):
    for k in range(len(o)):
      o[k] = instanciate_vars(o[k])
  elif isinstance(o, tuple):
    o = tuple(instanciate_vars(e) for e in o)
  elif isinstance(o, dict):
    for k in o:
      o[k] = instanciate_vars(o[k])
  return o


class RETURNNConfig:
  PYTHON_CODE = textwrap.dedent("""\
                #!rnn.py

                ${REGULAR_CONFIG}

                locals().update(**config)

                ${EXTRA_PYTHON_CODE}
                """)

  def __init__(self, config, post_config, extra_python_code='', extra_python_hash=None):
    self.config             = config
    self.post_config        = post_config
    self.extra_python_code  = extra_python_code
    self.extra_python_hash  = extra_python_hash if extra_python_hash is not None else extra_python_code

  def get(self, key, default=None):
    if key in self.post_config:
      return self.post_config[key]
    return self.config.get(key, default)

  def write(self, path):
    config = self.config
    config.update(self.post_config)

    config = instanciate_vars(config)

    config_lines = []
    unreadable_data = {}

    pp = pprint.PrettyPrinter(indent=2, width=150)
    for k, v in sorted(config.items()):
      if pprint.isreadable(v):
        config_lines.append('%s = %s' % (k, pp.pformat(v)))
      elif isinstance(v, tk.Path):
        unreadable_data[k] = v

    if len(unreadable_data) > 0:
      config_lines.append('import json')
      json_data = json.dumps(unreadable_data).replace('"', '\\"')
      config_lines.append('config = json.loads("%s")' % json_data)
    else:
      config_lines.append('config = {}')

    python_code = string.Template(self.PYTHON_CODE).substitute({ 'REGULAR_CONFIG' : '\n'.join(config_lines),
                                                                 'EXTRA_PYTHON_CODE' : self.extra_python_code })
    with open(path, 'wt', encoding='utf-8') as f:
      f.write(python_code)

  def hash(self):
    return { 'returnn_config':       self.config,
             'extra_python_hash': self.extra_python_hash }


class WriteRETURNNConfigJob(Job):
  def __init__(self, returnn_config):
    assert isinstance(returnn_config, RETURNNConfig)

    self.returnn_config = returnn_config

    self.returnn_config_file = self.output_path('returnn.config')

  def tasks(self):
    yield Task('run', resume='run', mini_task=True)

  def run(self):
    self.returnn_config.write(self.returnn_config_file.get_path())

  @classmethod
  def hash(self, kwargs):
    return super().hash(kwargs['returnn_config'].hash())
