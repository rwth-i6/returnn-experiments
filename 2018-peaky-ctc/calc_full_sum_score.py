#!/usr/bin/env python
# run in Python 2, because the Switchboard setup currently requires that...

"""
Script to calculate different kind of scores, e.g. with full sum criterion.
To fill in missing numbers of the full sum discussion.
See also here:

http://www-i6.informatik.rwth-aachen.de/i6wiki/FullSumExperiments

Basically:

Interested in different kind of scores:
* full-sum criterion with softmax prior
* full-sum criterion with fixed Viterbi prior
* full-sum criterion with lstm_out_prior (if existing)
* full-sum criterion without prior
* CE criterion (log perplexity) with fixed Viterbi alignment
* peakyness (relative occurence) of silence = expected value of softmax = softmax prior
* peakyness (relative occurence) of argmax prior
* ...

For these models:
* Converged CE-trained model
* Converged full-sum-trained model

This script should calculate all these scores / info for any given model.
If ran, it should see what's missing, including deps, and calculate that (e.g. also softmax prior).

"""

from __future__ import print_function
import better_exchook
import os
import sys
import numpy
from argparse import ArgumentParser, RawTextHelpFormatter
from subprocess import check_output

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/crnn")


# Starting with config updates. These are all very specific for these Switchboard setups.

num_outputs = 9001

commonfiles = {
    "corpus": "/u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz",
    "features": "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle",
    "lexicon": "/u/tuske/work/ASR/switchboard/corpus/train.lex.v1_0_3.ci.gz",
    "alignment": "dependencies/tuske__2016_01_28__align.combined.train",
    "cart": "/u/tuske/work/ASR/switchboard/initalign/data/%s" % {9001: "cart-9000"}[num_outputs]
}

EpochSplit = 1  # full dataset

_cf_cache = {}

def cf(filename):
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    if check_output(["hostname"]).strip().decode("utf8") in ["cluster-cn-211", "sulfid", "kalium"]:
        print("use local file: %s" % filename)
        return filename  # for debugging
    cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn

def get_sprint_dataset(data):
    assert data in ["train", "cv"]
    epochSplit = {"train": EpochSplit, "cv": 1}

    # see /u/tuske/work/ASR/switchboard/corpus/readme
    # and zoltans mail https://mail.google.com/mail/u/0/#inbox/152891802cbb2b40
    files = {}
    files["config"] = "config/training.config"
    files["corpus"] = "/u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz"
    files["segments"] = "dependencies/seg_%s" % {"train":"train", "cv":"cv_head3000"}[data]
    files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle"
    files["lexicon"] = "/u/tuske/work/ASR/switchboard/corpus/train.lex.v1_0_3.ci.gz"
    files["alignment"] = "dependencies/tuske__2016_01_28__align.combined.train"
    files["cart"] = "/u/tuske/work/ASR/switchboard/initalign/data/%s" % {9001: "cart-9000"}[num_outputs]
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    estimated_num_seqs = {"train": 227047, "cv": 3000}  # wc -l segment-file

    # features: /u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.*
    args = [
    "--config=" + files["config"],
    lambda: "--*.corpus.file=" + cf(files["corpus"]),
    lambda: "--*.corpus.segments.file=" + cf(files["segments"]),
    "--*.corpus.segment-order-sort-by-time-length=true",
    "--*.state-tying.type=cart",
    lambda: "--*.state-tying.file=" + cf(files["cart"]),
    "--*.trainer-output-dimension=%i" % num_outputs,
    lambda: "--*.lexicon.file=" + cf(files["lexicon"]),
    lambda: "--*.alignment-cache-path=" + cf(files["alignment"]),
    lambda: "--*.feature-cache-path=" + cf(files["features"]),
    "--*.log-channel.file=/dev/null",
    "--*.window-size=1",
    "--*.trainer-output-dimension=%i" % num_outputs
    ]
    return {
    "class": "ExternSprintDataset", "sprintTrainerExecPath": "sprint-executables/nn-trainer",
    "sprintConfigStr": args,
    "partitionEpoch": epochSplit[data],
    "estimated_num_seqs": estimated_num_seqs[data] // (epochSplit[data] or 1)}

def parse_tdp_config(s):
    s = s.replace(" ", "").replace("\t", "")
    return ["--*.tdp.%s" % l.strip() for l in s.splitlines() if l.strip()]

def get_sprint_error_signal_proc_args():
    files = commonfiles.copy()
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    return [
        "--config=config/ctc.train.config",
        "--action=python-control",
        "--python-control-loop-type=python-control-loop",
        "--*.python-segment-order=false",
        "--*.extract-features=false",  # we don't need features
        lambda: "--*.corpus.file=" + cf(files["corpus"]),
        "--*.state-tying.type=cart",
        lambda: "--*.state-tying.file=" + cf(files["cart"]),
        lambda: "--*.lexicon.file=" + cf(files["lexicon"]),
        "--*.feature-cache-path=should-not-be-needed",
        "--*.alignment-cache-path=should-not-be-needed",
        "--*.prior-file=dependencies/prior-fixed-f32.xml",  # should also not be needed
        "--*.lexicon.normalize-pronunciation=true",
        "--*.transducer-builder-filter-out-invalid-allophones=true",
        "--*.fix-allophone-context-at-word-boundaries=true",
        "--*.allow-for-silence-repetitions=false",
        "--*.normalize-lemma-sequence-scores=true",
        "--*.number-of-classes=%i" % num_outputs,
        "--*.log-channel.file=/dev/null",
    ] + parse_tdp_config("""
*.loop                  = %(loop)f
*.forward               = %(forward)f
*.skip                  = infinity
*.exit                  = %(forward)f
entry-m1.forward        = 0
entry-m2.forward        = 0
entry-m1.loop           = infinity
entry-m2.loop           = infinity
silence.loop            = %(sloop)f
silence.forward         = %(sforward)f
silence.skip            = infinity
silence.exit            = %(sforward)f
""" % {
#"loop": -numpy.log(0.65), "forward": -numpy.log(0.35),
#"sloop": -numpy.log(0.97), "sforward": -numpy.log(0.03)
#"sloop": -numpy.log(0.65), "sforward": -numpy.log(0.35)
#"loop": 0, "forward": 0, "sloop": 0, "sforward": 0
"loop": -numpy.log(0.5), "forward": -numpy.log(0.5),
"sloop": -numpy.log(0.5), "sforward": -numpy.log(0.5)
})

sprint_loss_opts = {
    "sprintExecPath": "sprint-executables/nn-trainer",
    "sprintConfigStr": "config:get_sprint_error_signal_proc_args",
    "minPythonControlVersion": 4,
    "sprintControlConfig": {},
    "numInstances": 2  # For debugging, better set this to 1.
}


config_update = {
    "sil_label_idx": 9000, # very specific for this CART
    "task": "eval",
    "EpochSplit": EpochSplit,
    "train": "",  # no train data
    "dev": get_sprint_dataset("train"),
    "commonfiles": commonfiles,
    "get_sprint_dataset": get_sprint_dataset,
    "parse_tdp_config": parse_tdp_config,
    "get_sprint_error_signal_proc_args": get_sprint_error_signal_proc_args,
    "sprint_loss_opts": sprint_loss_opts,
    "log": None,  # no logging to file (i.e. just stdout)
    "log_batch_size": True,
    "tf_log_memory_usage": True,
    "max_seq_length": -1,
    "batch_size": 20000,
    "max_seqs": 200,
}



network_update = {
    "out_fullsum_prior": {  # will be in +log space
        "class": "variable",
        "shape": (num_outputs,),
        "add_time_axis": True, "trainable": False,
        "init": None  # set afterwards. load from file
    },

    # log-likelihood: combine out + logprior
    "out_fullsum_scores": {
        "class": "combine", "kind": "eval", "from": ["output", "out_fullsum_prior"],
        "eval": "safe_log(source(0)) * am_scale - source(1) * prior_scale",
        "eval_locals": {
            # Set these afterwards.
            "am_scale": None,  # e.g. 0.3
            "prior_scale": None  # e.g. am_scale * 0.5
            }
    },
    "out_fullsum_bw": {
      "class": "fast_bw", "align_target": "sprint",
      "sprint_opts": sprint_loss_opts,
      "tdp_scale": None,  # set later
      "from": ["out_fullsum_scores"]},

    "output_fullsum": {
        "class" : "copy", "loss" : "via_layer", "from" : ["output"],
        "loss_opts": {
            "loss_wrt_to_act_in": "softmax",
            "align_layer": "out_fullsum_bw"
        }}
}


def check_valid_prior(filename):
    from Util import load_txt_vector
    v = load_txt_vector(filename)
    v = numpy.array(v)
    assert v.ndim == 1
    assert all(v < 0.0), "log space assumed"
    v = numpy.exp(v)  # to std prob space
    tot = numpy.sum(v)
    assert numpy.isclose(tot, 1.0, atol=1e-4)


class Globals:
    engine = None
    config = None
    dataset = None

    setup_name = None
    setup_dir = None
    epoch = None

    @classmethod
    def get_output_prefix(cls):
       return "fullsum-scores/out.%s.ep%03i." % (cls.setup_name, cls.epoch)

    @classmethod
    def get_softmax_prior_filename(cls):
       return cls.get_output_prefix() + "softmax-prior.txt"

    @classmethod
    def get_fullsum_scores_filename(cls, prior, am_scale, prior_scale, tdp_scale):
       """
       :param str prior: e.g. "none", "softmax", "fixed"
       :param float am_scale:
       :param float prior_scale:
       :param float tdp_scale:
       """
       return cls.get_output_prefix() + "fullsum-scores.prior_%s.am_scale_%f.prior_scale_%f.tdp_scale_%f.txt" % (
         prior, am_scale, prior_scale, tdp_scale)


def get_softmax_prior():
    fn = Globals.get_softmax_prior_filename()
    if os.path.exists(fn):
        print("Existing softmax prior:", fn)
        return fn
    print("Calculate softmax prior and save to:", fn)
    Globals.config.set("output_file", fn)
    Globals.engine.compute_priors(dataset=Globals.dataset, config=Globals.config)
    return fn


def calc_fullsum_scores(meta):
    from Util import betterRepr
    fn = Globals.get_fullsum_scores_filename(**meta)
    if os.path.exists(fn):
        print("Existing fullsum scores filename:", fn)
        print("content:\n%s\n" % open(fn).read())
        return fn
    # We assume that we have updated/extended the network topology.
    assert "output_fullsum" in Globals.engine.network.layers
    # Run it, and collect stats.
    analyzer = Globals.engine.analyze(data=Globals.dataset, statistics=None)
    print("fullsum score:", analyzer.score["cost:output_fullsum"])
    print("Write all to:", fn)
    with open(fn, "w") as f:
        f.write(betterRepr({
          "scores": analyzer.score,
          "errors": analyzer.error,
          "stats": analyzer.stats,
          "num_frames": analyzer.num_frames_accumulated}))
    return fn


def main():
    argparser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    argparser.add_argument("--model", required=True, help="or config, or setup")
    argparser.add_argument("--epoch", required=True, type=int)
    argparser.add_argument("--prior", help="none, fixed, softmax (default: none)")
    argparser.add_argument("--prior_scale", type=float, default=1.0)
    argparser.add_argument("--am_scale", type=float, default=1.0)
    argparser.add_argument("--tdp_scale", type=float, default=1.0)
    args = argparser.parse_args()

    cfg_fn = args.model
    if "/" not in cfg_fn:
        cfg_fn = "config-train/%s.config" % cfg_fn
    assert os.path.exists(cfg_fn)
    setup_name = os.path.splitext(os.path.basename(cfg_fn))[0]
    setup_dir = "data-train/%s" % setup_name
    assert os.path.exists(setup_dir)
    Globals.setup_name = setup_name
    Globals.setup_dir = setup_dir
    Globals.epoch = args.epoch

    config_update["epoch"] = args.epoch
    config_update["load_epoch"] = args.epoch
    config_update["model"] = "%s/net-model/network" % setup_dir

    import rnn
    rnn.init(
        configFilename=cfg_fn,
        config_updates=config_update,
        extra_greeting="calc full sum score.")
    Globals.engine = rnn.engine
    Globals.config = rnn.config
    Globals.dataset = rnn.dev_data

    assert Globals.engine and Globals.config and Globals.dataset
    # This will init the network, load the params, etc.
    Globals.engine.init_train_from_config(config=Globals.config, dev_data=Globals.dataset)

    # Do not modify the network here. Not needed.
    softmax_prior = get_softmax_prior()

    prior = args.prior or "none"
    if prior == "none":
        prior_filename = None
    elif prior == "softmax":
        prior_filename = softmax_prior
    elif prior == "fixed":
        prior_filename = "dependencies/prior-fixed-f32.xml"
    else:
        raise Exception("invalid prior %r" % prior)
    print("using prior:", prior)
    if prior_filename:
        assert os.path.exists(prior_filename)
        check_valid_prior(prior_filename)

    print("Do the stuff...")
    print("Reinit dataset.")
    Globals.dataset.init_seq_order(epoch=args.epoch)

    network_update["out_fullsum_scores"]["eval_locals"]["am_scale"] = args.am_scale
    network_update["out_fullsum_scores"]["eval_locals"]["prior_scale"] = args.prior_scale
    network_update["out_fullsum_bw"]["tdp_scale"] = args.tdp_scale
    if prior_filename:
        network_update["out_fullsum_prior"]["init"] = "load_txt_file(%r)" % prior_filename
    else:
        network_update["out_fullsum_prior"]["init"] = 0
    from copy import deepcopy
    Globals.config.typed_dict["network"] = deepcopy(Globals.config.typed_dict["network"])
    Globals.config.typed_dict["network"].update(network_update)
    # Reinit the network, and copy over params.
    from Pretrain import pretrainFromConfig
    pretrain = pretrainFromConfig(Globals.config)  # reinit Pretrain topologies if used
    if pretrain:
      new_network_desc = pretrain.get_network_json_for_epoch(Globals.epoch)
    else:
      new_network_desc = Globals.config.typed_dict["network"]
    assert "output_fullsum" in new_network_desc
    print("Init new network.")
    Globals.engine.maybe_init_new_network(new_network_desc)

    print("Calc scores.")
    calc_fullsum_scores(
      meta=dict(prior=prior, prior_scale=args.prior_scale, am_scale=args.am_scale, tdp_scale=args.tdp_scale))

    rnn.finalize()
    print("Bye.")


if __name__ == "__main__":
    better_exchook.install()
    main()
