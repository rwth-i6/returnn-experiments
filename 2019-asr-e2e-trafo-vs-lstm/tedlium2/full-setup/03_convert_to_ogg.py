#!/usr/bin/env python3

import stm_reader
import os
from subprocess import check_call
from argparse import ArgumentParser

BaseDir = "data/dataset-extracted/TEDLIUM_release2"
DestDir = "data/dataset"


def handle_part(name, keep_existing_ogg):
  """
  :param str name: "train", "dev" or "test"
  :param bool keep_existing_ogg:
  """
  dirname = "%s/%s/stm" % (BaseDir, name)
  assert os.path.isdir(dirname)
  dest_dirname = "%s/%s" % (DestDir, name)
  dest_meta_filename = "%s/%s.txt" % (DestDir, name)
  dest_meta_file = open(dest_meta_filename, "w")
  dest_meta_file.write("[\n")
  os.makedirs(dest_dirname, exist_ok=True)
  for seq in stm_reader.read_stm_dir(dirname):
    sph_filename = "%s/%s/sph/%s.sph" % (BaseDir, name, seq.speaker)
    assert os.path.isfile(sph_filename)
    assert seq.start < seq.end
    duration = seq.end - seq.start
    assert duration > 0
    dest_filename = "%s/%s_%s_%s.ogg" % (dest_dirname, seq.speaker, seq.start, seq.end)
    if os.path.exists(dest_filename) and keep_existing_ogg:
      print("already exists, skip: %s" % os.path.basename(dest_filename))
    else:
      if os.path.exists(dest_filename):
        print("already exists, delete: %s" % os.path.basename(dest_filename))
        os.remove(dest_filename)
      cmd = ["ffmpeg", "-i", sph_filename, "-ss", str(seq.start), "-t", str(duration), dest_filename]
      print("$ %s" % " ".join(cmd))
      check_call(cmd)
    dest_meta_file.write("{'text': %r, 'tags': %r, 'file': %r, 'duration': %s},\n" % (seq.text, seq.tags, os.path.basename(dest_filename), duration))
  dest_meta_file.write("]\n")
  dest_meta_file.close()


def print_stats(name):
  """
  :param str name: "train", "dev" or "test"
  """
  print("%s:" % name)
  filename = "%s/%s.txt" % (DestDir, name)
  assert os.path.isfile(filename)
  data = eval(open(filename).read())
  assert isinstance(data, list)
  print("  num seqs:", len(data))
  total_duration = 0.0
  total_num_chars = 0
  for seq in data:
    total_duration += seq["duration"]
    total_num_chars += len(seq["text"])
  print("  total duration:", total_duration)
  print("  total num chars:", total_num_chars)


def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--keep_existing_ogg", action="store_true")
  arg_parser.add_argument("--stats_only", action="store_true")
  args = arg_parser.parse_args()
  assert os.path.isdir(BaseDir)
  if not args.stats_only:
    os.makedirs(DestDir, exist_ok=True)
    handle_part("train", keep_existing_ogg=args.keep_existing_ogg)
    handle_part("dev", keep_existing_ogg=args.keep_existing_ogg)
    handle_part("test", keep_existing_ogg=args.keep_existing_ogg)
  print_stats("train")
  print_stats("dev")
  print_stats("test")


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
