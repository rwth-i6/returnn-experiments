#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import os
from zipfile import ZipFile
from glob import glob

zip_dir = "data/dataset"
zip_files = ["train-clean-100.zip", "train-clean-360.zip", "train-clean-500.zip"]
zip_files = ["%s/%s" % (zip_dir, fn) for fn in zip_files]
assert all([os.path])

for fn in sorted(glob("train-*/*/*/*.trans.txt")):
    for l in open(fn).read().splitlines():
        seq_name, txt = l.split(" ", 1)
        print(txt)

