#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import argparse
from glob import glob
import os
import sys
import re

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment", default="returnn")
argparser.add_argument("--beam_size", default=12, type=int)
argparser.add_argument("--print_epoch_only", action="store_true")
args = argparser.parse_args()
exp_dir = "data/exp-%s" % args.experiment
assert os.path.exists(exp_dir)

datasets = ["dev-clean", "dev-other", "test-clean", "test-other"]
dev_dataset = "dev-other"
# Files via tools/search.py script.
files = glob("%s/search.%s.*.beam%i.recog.scoring.wer" % (exp_dir, dev_dataset, args.beam_size))
assert files, "no recog done yet?"
files_and_scores = [(float(open(fn).read()), fn) for fn in files]

if args.print_epoch_only:
  fn = sorted(files_and_scores)[0][1]
  m1 = re.match(".*/search\\.%s\\.ep([0-9]+)\\.beam.*" % dev_dataset, fn)
  epoch = int(m1.group(1))
  print(epoch)
  sys.exit()

print("Experiment %r: Found %i recogs with beam size %i." % (args.experiment, len(files), args.beam_size))
print()

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
