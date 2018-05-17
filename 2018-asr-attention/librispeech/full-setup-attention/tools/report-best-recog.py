#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import argparse
from glob import glob
import os
import re

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment", default="returnn")
args = argparser.parse_args()
exp_dir = "data/exp-%s" % args.experiment
assert os.path.exists(exp_dir)

datasets = ["dev-clean", "dev-other", "test-clean", "test-other"]
dev_dataset = "dev-other"
# Files via tools/search.py script.
files = glob("%s/search.%s.*.recog.scoring.wer" % (exp_dir, dev_dataset))
assert files, "no recog done yet?"
files_and_scores = [(float(open(fn).read()), fn) for fn in files]

for i, (score, fn) in enumerate(sorted(files_and_scores)[:2]):
  print("%i.-best recog by dataset %s, %s%% WER:" % (i + 1, dev_dataset, score))
  # fn is e.g. ".../search.dev-clean.ep80.beam12.recog.scoring.wer"
  m1 = re.match(".*/search\\.%s\\.ep([0-9]+)\\.beam.*" % dev_dataset, fn)
  m2 = re.match(".*/search\\.%s\\.(.*)$" % dev_dataset, fn)
  assert m1 and m2
  epoch = int(m1.group(1))
  fn_postfix = m2.group(1)
  print("Epoch: %i" % epoch)
  for dataset in datasets:
    try:
      print("%s: %s%% WER" % (dataset, float(open("%s/search.%s.%s" % (exp_dir, dataset, fn_postfix)).read())))
    except Exception as e:
      print("%s: %s" % (dataset, e))
  print()
