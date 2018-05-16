#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import os
from zipfile import ZipFile, ZipInfo
from glob import glob

zip_dir = "."  # run this from the right path
zip_files = ["train-clean-100.zip", "train-clean-360.zip", "train-other-500.zip"]
zip_files = ["%s/%s" % (zip_dir, fn) for fn in zip_files]
assert all([os.path.exists(fn) for fn in zip_files])
zip_files = [ZipFile(fn) for fn in zip_files]

for zip_file in zip_files:
  assert zip_file.filelist
  assert zip_file.filelist[0].filename.startswith("LibriSpeech/")
  for info in zip_file.filelist:
    assert isinstance(info, ZipInfo)
    path = info.filename.split("/")
    assert path[0] == "LibriSpeech", "does not expect %r (%r)" % (info, info.filename)
    if path[1].startswith("train-"):
      subdir = path[1]  # e.g. "train-clean-100"
      if path[-1].endswith(".trans.txt"):
        for l in zip_file.read(info).decode("utf8").splitlines():
          seq_name, txt = l.split(" ", 1)
          print(txt)
