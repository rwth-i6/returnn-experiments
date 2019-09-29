#!/usr/bin/env python3

import better_exchook
better_exchook.install()

import argparse
import sys
import os
import time
from subprocess import check_output

default_python_bin = "python3"
returnn_dir_name = "returnn"


def run(args, **kwargs):
    import subprocess
    kwargs = kwargs.copy()
    print("$ %s" % " ".join(args), {k: v if k != "input" else "..." for (k, v) in kwargs.items()})
    try:
        subprocess.run(args, **kwargs, check=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)


argv = sys.argv[1:]
if "--" in argv:
    argv, returnn_argv = argv[:argv.index("--")], argv[argv.index("--") + 1:]
else:
    returnn_argv = []

argparser = argparse.ArgumentParser()
argparser.add_argument("model", help="'returnn' if the train config was 'return.config'")
argparser.add_argument("epoch", type=int)
argparser.add_argument("--data", default="dev-clean", help="dev-clean, dev-other, test-clean, test-other")
argparser.add_argument("--beam_size", type=int, default=12)
argparser.add_argument("--device", default="gpu")
argparser.add_argument("--use_existing", action="store_true")
argparser.add_argument("--allow_tmp", action="store_true")
argparser.add_argument("--out_dir")
argparser.add_argument("--recog_prefix")
argparser.add_argument("--search_output_layer", default="decision")
args = argparser.parse_args(argv)

start_time = time.time()

config_fn = "%s.config" % args.model
assert os.path.exists(config_fn)

out_dir = "data/exp-%s" % args.model
if args.out_dir:
    out_dir = args.out_dir
if args.recog_prefix:
    recog_prefix = "%s/%s" % (out_dir, args.recog_prefix)
else:
    recog_prefix = "%s/search.%s.ep%i.beam%i.recog" % (out_dir, args.data, args.epoch, args.beam_size)
recog_bpe_file = "%s.bpe" % recog_prefix
recog_words_file = "%s.words" % recog_prefix
recog_wer_file = "%s.scoring.wer" % recog_prefix


if os.path.exists(recog_wer_file):
    print("Final recog WER file already exists:", recog_wer_file)
    print("Exiting now. Please delete that file to rerun.")
    sys.exit()


def check_recog_bpe_file():
    with open(recog_bpe_file, "w") as f:
        f.close()
    os.remove(recog_bpe_file)


if args.use_existing:
    assert os.path.exists(recog_bpe_file), "--use_existing but file does not exist"
    print("Using existing file: %s" % recog_bpe_file)

else:
    check_recog_bpe_file()

    run([
        default_python_bin,
        "%s/rnn.py" % returnn_dir_name, config_fn, "++load_epoch", "%i" % args.epoch,
        "++device", args.device,
        "--task", "search", "++search_data", "config:get_dataset(%r)" % args.data,
        "++beam_size", "%i" % args.beam_size,
        "++need_data", "False",  # the standard datasets (train, dev in config) are not needed to be loaded
        "++max_seq_length", "0",
        "++search_output_file", os.path.abspath(recog_bpe_file),
        "++search_output_file_format", "py",
        "++search_do_eval", "0",
        "++search_output_layer", args.search_output_layer,
        ] +
        returnn_argv)

    assert os.path.exists(recog_bpe_file)

if os.path.exists(recog_words_file):
    os.remove(recog_words_file)
run(["tools/search-bpe-to-words.py", recog_bpe_file, "--out", recog_words_file])

run([
    default_python_bin,
    "%s/tools/calculate-word-error-rate.py" % returnn_dir_name,
    "--expect_full",
    "--hyps", recog_words_file,
    "--refs", "data/dataset/%s.trans.raw" % args.data,
    "--out", recog_wer_file,
    ])

print("elapsed time: %s" % (time.time() - start_time))
