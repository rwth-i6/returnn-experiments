#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import argparse
import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir) + "/returnn"
assert os.path.exists(returnn_dir)
sys.path.append(returnn_dir)
log = sys.stderr

from Config import Config

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment", default="returnn", help="will look in data/exp-<experiment> for scores/models")
args = argparser.parse_args()

exp_config_fn = "%s.config" % args.experiment
assert os.path.exists(exp_config_fn)
config = Config()
config.load_file(exp_config_fn)
num_epochs = config.int("num_epochs", 0)
assert num_epochs > 0, "no num_epochs in %r" % exp_config_fn
print("Experiment config: %s" % (exp_config_fn,), file=log)

exp_dir = "data/exp-%s" % args.experiment
assert os.path.exists(exp_dir)
train_scores_fn = "%s/train-scores.data" % exp_dir
assert os.path.exists(train_scores_fn), "Train-scores file not found. Maybe no epoch fully trained yet?"

# nan/inf, for some broken newbob.data
nan = float("nan")
inf = float("inf")

# simple wrapper, to eval newbob.data
def EpochData(learningRate, error):
    d = {}
    d["learning_rate"] = learningRate
    d.update(error)
    return d

train_scores_data = open(train_scores_fn).read()
try:
    # train_scores: dict epoch -> dict key (dev_score_output or so) -> value
    train_scores = eval(train_scores_data)
except Exception as e:
    print("%s: Train-scores eval exception: %r" % (train_scores_fn, e), file=log)
    raise

last_epoch = max(train_scores.keys())
print("Trained epochs: %i/%i" % (last_epoch, num_epochs), file=log)
keys = sorted(set(sum([list(info.keys()) for info in train_scores.values()], [])))
if not any([key.startswith("dev_") for key in keys]):
    print("No cross validation (on dev) done (yet). Keys: %r" % (keys,), file=log)
    sys.exit(1)
finished = False
if last_epoch >= num_epochs:
    if any([key.startswith("dev_") for key in train_scores[last_epoch].keys()]):
        finished = True
print("Finished training: %r" % finished, file=log)


ds = {}  # ep -> dict with more...

def add_suggest(ep, temp=None, reason=None):
    if ep in ds: return
    ds[ep] = {"epoch": ep, "temporary_suggestion": temp, "reason": reason}

if finished:
    add_suggest(last_epoch, temp=False, reason="last epoch")

n = 5
while n <= last_epoch:
    if n * 4 >= num_epochs:
        add_suggest(n, temp=False, reason="intermediate progress")
    n *= 2

# collect suggestions based on dev scores
for score_key in keys:
    if not score_key.startswith("dev_"):
        continue
    dev_scores = []  # (value, ep) list, for this key
    for epoch, info in sorted(train_scores.items()):
        if score_key not in info:
            continue
        dev_scores += [(info[score_key], epoch)]
    assert dev_scores
    dev_scores.sort()
    if dev_scores[0][0] == dev_scores[-1][0]:
        # All values are the same (e.g. 0.0), so no information. Just ignore this score_key.
        continue
    if dev_scores[0] == (0.0, 1):
        # Heuristic. Ignore the key if it looks invalid.
        continue
    for i, (_, ep) in enumerate(sorted(dev_scores)[:2]):
        add_suggest(ep, temp=not finished, reason="%i.-best on %r" % (i + 1, score_key))

print("Suggested epochs:", file=log)
return_eps = []
for epoch, d in sorted(ds.items()):
    print(d, file=log)
    if not d["temporary_suggestion"]:
        return_eps.append(epoch)

# Final output on stdout.
print(" ".join(map(str, return_eps)))
