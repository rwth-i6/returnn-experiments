import gzip
import os
import shutil
import subprocess as sp
import xml.etree.ElementTree as ET

from sisyphus import *
Path = setup_path(__package__)
Variable = tk.Variable


class MultiPath:
  def __init__(self, path_template, hidden_paths, cached=False, path_root=None, hash_overwrite=None):
    self.path_template  = path_template
    self.hidden_paths   = hidden_paths
    self.cached         = cached
    self.path_root      = path_root
    self.hash_overwrite = hash_overwrite

  def __str__(self):
    if self.path_root is not None:
      result = os.path.join(self.path_root, self.path_template)
    else:
      result = self.path_template
    if self.cached:
      result = '`cf %s`' % result
    return result

  def __sis_state__(self):
    return { 'path_template': self.path_template if self.hash_overwrite is None else self.hash_overwrite,
             'hidden_paths' : self.hidden_paths,
             'cached'       : self.cached         }


class MultiOutputPath(MultiPath):
  def __init__(self, creator, path_template, hidden_paths, cached=False):
    super().__init__(os.path.join(creator._sis_path(gs.JOB_OUTPUT), path_template), hidden_paths, cached, gs.BASE_DIR)


def write_paths_to_file(file, paths):
  with open(tk.uncached_path(file), 'w') as f:
    for p in paths:
      f.write(tk.uncached_path(p) + '\n')


def zmove(src, target):
  src = tk.uncached_path(src)
  target = tk.uncached_path(target)

  if not src.endswith('.gz'):
    tmp_path = src + '.gz'
    if os.path.exists(tmp_path):
      os.unlink(tmp_path)
    sp.check_call(['gzip', src])
    src += '.gz'
  if not target.endswith('.gz'):
    target += '.gz'

  shutil.move(src, target)


def delete_if_exists(file):
  if os.path.exists(file):
    os.remove(file)


def delete_if_zero(file):
  if os.path.exists(file) and os.stat(file).st_size == 0:
    os.remove(file)


def backup_if_exists(file):
  if os.path.exists(file):
    dir, base = os.path.split(file)
    base = add_suffix(base, '.gz')
    idx = 1
    while os.path.exists(os.path.join(dir, 'backup.%.4d.%s' % (idx, base))):
      idx += 1
    zmove(file, os.path.join(dir, 'backup.%.4d.%s' % (idx, base)))


def remove_suffix(string, suffix):
  if string.endswith(suffix):
    return string[:-len(suffix)]
  return string


def add_suffix(string, suffix):
  if not string.endswith(suffix):
    return string + suffix
  return string


def partition_into_tree(l, m):
  """ Transforms the list l into a nested list where each sub-list has at most length m + 1"""
  nextPartition = partition = l
  while len(nextPartition) > 1:
    partition = nextPartition
    nextPartition = []
    d = len(partition) // m
    mod = len(partition) % m
    if mod <= d:
      p = 0
      for i in range(mod):
        nextPartition.append(partition[p:p + m + 1])
        p += m + 1
      for i in range(d - mod):
        nextPartition.append(partition[p:p + m])
        p += m
      assert p == len(partition)
    else:
      p = 0
      for i in range(d):
        nextPartition.append(partition[p:p + m])
        p += m
      nextPartition.append(partition[p:p + mod])
      assert p + mod == len(partition)
  return partition


def reduce_tree(func, tree):
  return func([(reduce_tree(func, e) if type(e) == list else e) for e in tree])


def uopen(path, *args, **kwargs):
  path = tk.uncached_path(path)
  if path.endswith('.gz'):
    return gzip.open(path, *args, **kwargs)
  else:
    return open(path, *args, **kwargs)


def get_val(var):
  if isinstance(var, Variable):
    return var.get()
  return var


def chunks(l, n):
  """
  :param list[T] l: list which should be split into chunks
  :param int n: number of chunks
  :return: yields n chunks
  :rtype: list[list[T]]
  """
  bigger_count = len(l) % n
  start = 0
  block_size = len(l) // n
  for i in range(n):
    end = start + block_size + (1 if i < bigger_count else 0)
    yield l[start:end]
    start = end
