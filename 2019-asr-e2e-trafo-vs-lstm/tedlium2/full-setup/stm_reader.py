
from collections import namedtuple
from decimal import Decimal
import re
import sys
import os
from glob import glob

StmSeq = namedtuple("StmSeq", ["speaker", "start", "end", "tags", "text"])
StmSeqRegExpPattern = "^([A-Za-z0-9_.]+) 1 ([A-Za-z0-9_.]+) ([0-9.]+) ([0-9.]+) <([A-Za-z0-9,_]+)> ([\\w' ]+)$"
StmSeqRegExp = re.compile(StmSeqRegExpPattern)


def parse_stm_seq(line):
  """
  :param str line:
  :rtype: StmSeq|None
  """
  m = StmSeqRegExp.match(line)
  if not m:
    m2 = re.match(StmSeqRegExpPattern[:-1], line)
    raise Exception("line %r, no match to %r. but prefix: %r" % (line, StmSeqRegExp, line[:m2.end()] if m2 else None))
  name1, name2, start, end, tags, text = m.groups()
  if text == "ignore_time_segment_in_scoring":
    return None
  # assert name1 == name2, "line %r" % line  # most often, but not always?
  start, end = Decimal(start), Decimal(end)
  return StmSeq(speaker=name1, start=start, end=end, tags=tags, text=text)


def read_stm(filename):
  """
  :param str filename:
  :rtype: yields StmSeq
  """
  lines = open(filename).read().splitlines()
  # Examples:
  # AaronHuey_2010X 1 AaronHuey_2010X 16.13 24.16 <o,f0,female> i'm here today to show my photographs of the lakota ...
  # AaronHuey_2010X 1 AaronHuey_2010X 520.18 522.81 <o,f0,female> when i saw them with eyes still young
  # XDRTB_2008 1 XDRTB_2008 88.55 89.57 <o,f0,male> from the halls of
  for line in lines:
    seq = parse_stm_seq(line)
    if not seq:
      continue
    yield seq


def read_stm_dir(dirname):
  """
  :param str dirname:
  :rtype: yields StmSeq
  """
  files = glob(dirname + "/*.stm")
  assert files, "no stm files in %r found" % dirname
  for fn in files:
    yield from read_stm(fn)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  arg = sys.argv[1]
  if os.path.isfile(arg):
    for seq in read_stm(arg):
      print(seq)
  elif os.path.isdir(arg):
    for seq in read_stm_dir(arg):
      print(seq)
  else:
    raise Exception("%r is not a file nor dir" % arg)
