#!/usr/bin/env python3

from __future__ import print_function
import better_exchook
import os
import sys
import re
import numpy
from argparse import ArgumentParser, RawTextHelpFormatter
from subprocess import check_output

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/tools-multisetup")

import tools
from tools import Settings


models = [
  "dropout01.l2_1e_2.5l.n500.max_seqs100.grad_noise03.nadam.lr05e_3.nbm6.nbrl.grad_clip_inf.nbm3",
  "dropout01.l2_1e_2.6l.n500.inpstddev3.fl2.max_seqs100.grad_noise03.nadam.lr05e_3.nbm6.nbrl.grad_clip_inf.nbm3",
  "ctcfbw.p2ff500c.ce05s01.prior05am.cpea001.am01b03.nopretrain.nbf",
  "ctcfbw.p2ff500c.noprior",
  "ctcfbw.p2ff500c.nobias",
  "ctcfbw.p2ff500c.noprior.nobias",
  "ctcfbw.noff.noprior",
  "ctcfbw.noff.noprior.nobias",
  "ctcfbw.6l",
  "ctcfbw.6l.nobias",
  "ctcfbw.6l.noprior"
]

num_labels = 9001
sil_label_id = 9000


variants = [
  {"prior": "none"},
  {"prior": "softmax"},
  {"prior": "fixed"},
  {"prior": "none", "tdp_scale": 0},
  {"prior": "softmax", "tdp_scale": 0},
  {"prior": "fixed", "tdp_scale": 0},
]


def run(args, **kwargs):
  import subprocess
  kwargs = kwargs.copy()
  print("$ %s" % " ".join(args), {k: v if k != "input" else "..." for (k, v) in kwargs.items()})
  try:
    subprocess.run(args, **kwargs, check=True)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)


qsub_opts = [
  "-notify", "-hard",
  "-l", "h_vmem=15G",
  "-l", "h_rt=10:00:00",
  "-l", "gpu=1",
  "-l", "qname=*1080*|*TITAN*",
  "-l", "num_proc=4"]

# echo "exec /u/zeyer/environments/setup/common/bin/qint.py recog.q.sh recog.pass1.init --execute --guard-level 3" | qsub
# -N recog.pass1.init -cwd -S /bin/bash -hard -l h_vmem=8950M -l h_rt=2:59:00 -j yes -o qdir/q.log

def qsub_name_from_args(args):
  return "qsub_" + "_".join(args).replace("./", "").replace("/", "").replace(" ", "")

def qsub(args):
  name = qsub_name_from_args(args)
  run(
    ["qsub", "-cwd", "-S", "/bin/bash", "-j", "yes", "-o", "fullsum-scores"] + qsub_opts + ["-N", name],
    input=" ".join(args).encode("utf8"))



r_epoch = re.compile('epoch *([0-9]+)')


def get_wers(fn):
    wers = {}  # epoch -> wer
    for l in open(fn).read().splitlines():
        k, v = l.split(":", 1)
        epoch = r_epoch.match(k).group(1)
        wers[int(epoch)] = float(v)
    return wers


def get_best_epoch(model):
    fn = "scores/%s.recog.%ss.txt" % (model, Settings.recog_metric_name)
    assert os.path.exists(fn)
    wers = get_wers(fn)
    return sorted([(score, ep) for (ep, score) in wers.items()], reverse=not Settings.recog_score_lower_is_better)[0]


def get_train_scores(train_scores_file):
    train_scores = {}  # key -> ep -> score
    for l in open(train_scores_file).read().splitlines():
        m = re.match('epoch +([0-9]+) ?(.*): *(.*)', l)
        if not m:
            #print("warning: no match for %r" % l)
            continue
        ep, key, value = m.groups()
        if "error" in key or "score" in key or not key:
            train_scores.setdefault(key, {})[int(ep)] = float(value)
    return train_scores


def open_res(fn):
  txt = open(fn).read()
  # sth like "<Util.Stats instance at 0x7f8a607ad248>" in it
  txt = re.sub("<.*>", "None", txt)
  # simpler to not import NumbersDict
  txt = re.sub("NumbersDict\\({", "({", txt)
  try:
    d = eval(txt)
  except Exception as exc:
    print("Parse exception:", exc)
    print("txt:")
    print(txt)
    raise
  assert isinstance(d, dict)
  return d


def check_sge_job_exists(args):
  name = qsub_name_from_args(args)
  from subprocess import Popen, DEVNULL
  p = Popen(["qstat", "-j", name], stdout=DEVNULL, stderr=DEVNULL)
  ret = p.wait()
  return ret == 0


def main():
  argparser = ArgumentParser()
  argparser.add_argument("--calc", help="none, local or sge")
  args = argparser.parse_args()
  #qsub(["echo", "test hello world"])
  #sys.exit()

  for model in models:
    score, ep = get_best_epoch(model)
    print("model %s, best epoch: %s" % (model, ep))
    print("  WER (dev): %.1f%%" % score)
    train_scores_fn = "scores/%s.train.info.txt" % model
    assert os.path.exists(train_scores_fn)
    train_scores = get_train_scores(train_scores_fn)
    dev_err_key = "dev_error"
    if dev_err_key not in train_scores:
      dev_err_key = "dev_error_output"
    print("  FER (cv): %.1f%%" % (train_scores[dev_err_key][ep] * 100.))
    prefix = "fullsum-scores/out.%s.ep%03i." % (model, ep)
    sm_prior_fn = prefix + "softmax-prior.txt"
    print("  sm prior exists:", os.path.exists(sm_prior_fn))
    if os.path.exists(sm_prior_fn):
      sm_prior = numpy.loadtxt(sm_prior_fn)
      assert sm_prior.shape == (num_labels,)
      print("    sm prior sil label prob: %.1f%%" % (numpy.exp(sm_prior[sil_label_id]) * 100.))
    res = None
    for variant in variants[:len(variants) if os.path.exists(sm_prior_fn) else 1]:
      prior = variant["prior"]
      tdp_scale = variant.get("tdp_scale", 1.0)
      am_scale = variant.get("am_scale", 1.0)
      prior_scale = variant.get("prior_scale", 1.0)
      res_fn = prefix + "fullsum-scores.prior_%s.am_scale_%f.prior_scale_%f.tdp_scale_%f.txt" % (
        prior, float(am_scale), float(prior_scale), float(tdp_scale))
      print("  variant %r exists:" % variant, os.path.exists(res_fn))
      if not os.path.exists(res_fn):
        cmd_args = [
            "./calc_full_sum_score.py", "--model", model, "--epoch", str(ep),
            "--prior", prior,
            "--tdp_scale", str(tdp_scale),
            "--am_scale", str(am_scale),
            "--prior_scale", str(prior_scale),
            ]
        if not args.calc:
          print("    not calculating. SGE job exists:", check_sge_job_exists(cmd_args))
        elif args.calc == "local":
          run(cmd_args)
        elif args.calc == "sge":
          if check_sge_job_exists(cmd_args):
            print("    SGE job already exists")
          else:
            qsub(cmd_args)
        else:
          raise Exception("invalid calc %r" % args.calc)
      else:  # it exists
        res = open_res(res_fn)
        print("    fullsum score: %.3f" % res["scores"]["cost:output_fullsum"])
    # Some more stats are there. However, hardcoded from the first output layer, sometimes the FF layer, i.e. not interesting.
    #if res:
    #  print("    CE score: %.3f" % res["stats"]["stats:analyze:accumulated_loss_ce"])


if __name__ == "__main__":
  main()

