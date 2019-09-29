#!/usr/bin/env python3

import os
from zipfile import ZipFile, ZipInfo
from glob import glob
from argparse import ArgumentParser


def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("file")
  args = arg_parser.parse_args()
  assert os.path.exists(args.file)
  name, ext = os.path.splitext(os.path.basename(args.file))
  assert ext == ".zip"
  zip_file = ZipFile(args.file)

  data = eval(zip_file.open("%s.txt" % name).read())
  for seq in data:
    print(seq["text"])


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
