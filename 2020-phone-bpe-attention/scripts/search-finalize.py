#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import argparse
import sys
import os
import time
from subprocess import check_output

my_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "%s/tools-multisetup" % my_dir)
sys.path.insert(1, "%s/crnn" % my_dir)

import tools
from Config import Config  # returnn


def run(args, **kwargs):
    import subprocess
    kwargs = kwargs.copy()
    print("$ %s" % " ".join(args), {k: v if k != "input" else "..." for (k, v) in kwargs.items()})
    try:
        subprocess.run(args, **kwargs, check=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)


argparser = argparse.ArgumentParser()
argparser.add_argument("model")
argparser.add_argument("epoch", type=int)
argparser.add_argument("--data", default="dev", help="cv, dev, hub5e_01, rt03s")
argparser.add_argument("--out_dir", default=".")
argparser.add_argument("--recog_prefix")
args = argparser.parse_args()

config_fn = "config-train/%s.config" % args.model
assert os.path.exists(config_fn)

config = Config()
config.load_file(config_fn)

out_dir = args.out_dir
if args.recog_prefix:
    recog_prefix = "%s/%s" % (out_dir, args.recog_prefix)
else:
    recog_prefix = "%s/%s" % (out_dir, "scoring-%s" % args.data)

recog_bpe_file = "%s.bpe" % recog_prefix
recog_words_file = "%s.words" % recog_prefix
recog_ctm_file = "%s.ctm" % recog_prefix

# Maybe from an earlier run, these exists.
if os.path.exists(recog_words_file):
    os.remove(recog_words_file)
if os.path.exists(recog_ctm_file):
    os.remove(recog_ctm_file)

#run(["tools-recog/search-bpe-to-words.py", recog_bpe_file, "--out", recog_words_file])

words_postprocess_func = config.typed_dict.get("words_postprocess", None)
if words_postprocess_func:
    print("run words_postprocess %r from config" % words_postprocess_func)
    new_recog_words_file = words_postprocess_func(words_filename=recog_bpe_file)
    assert os.path.exists(new_recog_words_file)
    recog_words_file = new_recog_words_file

run(["tools-recog/search-words-to-ctm.py", recog_words_file, "--out", recog_ctm_file,
     "--corpus", "/u/tuske/work/ASR/switchboard/corpus/xml/%s.corpus.gz" % args.data])

# Should put data into scoring/ directory.
run(["./tools-recog/score.%s.sh" % args.data, recog_ctm_file], cwd=os.path.dirname(recog_ctm_file))
