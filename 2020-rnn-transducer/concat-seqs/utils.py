
from sisyphus.tools import sis_hash


def generic_open(filename, mode="r"):
  """
  Wrapper around :func:`open`.
  Automatically wraps :func:`gzip.open` if filename ends with ``".gz"``.

  :param str filename:
  :param str mode: text mode by default
  :rtype: typing.TextIO|typing.BinaryIO
  """
  if filename.endswith(".gz"):
    import gzip
    if "b" not in mode:
      mode += "t"
    return gzip.open(filename, mode)
  return open(filename, mode)


def hash_limited_len_name(name, limit=200):
  """
  :param str name:
  :param int limit:
  :return: name, maybe truncated (by hash) such that its len (in bytes) is <=200
  :rtype: str
  """
  name_b = name.encode("utf8")
  if len(name_b) < limit:
    return name
  assert len(name_b) == len(name)  # ascii expected currently...
  h = sis_hash(name_b)
  name = "%s...%s" % (name[:limit - 3 - len(h)], h)
  assert len(name) == limit
  return name
