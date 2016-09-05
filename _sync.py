#!/usr/bin/env python3

"""
Copies the selected configs over from my environment to this directory.
This will only make sense to be run from inside our chair.
"""

try:
	import better_exchook
	better_exchook.install()
except ImportError:
	pass

import os, sys, re
import shutil
mydir = os.path.dirname(__file__) or "."
relmydir = os.path.relpath(mydir)
if relmydir[:2] != "..": mydir = relmydir

base_files = [
	"train.q.sh",
	"theano-cuda-activate.sh",
	"config/training.config",
	"config/setup.base.config",
	"config/network.config",
	"config/shared/lexicon.config",
	"config/shared/common.config",
	"config/shared/corpus.config",
	"config/gt.config",
	"flow/accumulation.flow",
	"flow/concat.1x.flow",
	"flow/base.cache.flow",
]

quaero_dst_base_dir = mydir + "/quaero-en50h"
quaero_src_base_dir = "/u/zeyer/setups/quaero-en/training/quaero-train11/50h/ann/2015-07-29--lstm-gt50"
quaero_experiments = [
	"dropout01.l2_1e_2.1l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.2l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.4l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.5l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.6l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.7l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.8l.n500.max_seqs40.adam.lr1e_3",

	"dropout01.l2_1e_2.5l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.5l.n600.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.5l.n700.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.5l.n800.max_seqs40.adam.lr1e_3",

	"dropout01.3l.n300",
	"dropout01.3l.n500",
	"dropout01.3l.n700",
	"dropout01.3l.n1000",

	"dropout01.3l.n500.sgd.lr1e_3",
	"dropout01.3l.n500.sgd.lr1e_4",
	"dropout01.3l.n500.sgd.lr1e_4.momentum09",
	"dropout01.3l.n500.sgd.lr1e_4.momentum2_09",
	"dropout01.3l.n500.sgd.lr1e_4.momentum2_05",
	"dropout01.3l.n500.sgd.lr1e_4.nesterov09",
	"dropout01.3l.n500.sgd.lr05e_4",
	"dropout01.3l.n500.sgd.lr05e_4.momentum2_09",
	"dropout01.3l.n500.sgd.lr1e_5",
	"dropout01.3l.n500.sgd.lr1e_5_1e_4",
	"dropout01.3l.n500.mnsgd.lr1e_4",
	"dropout01.3l.n500.mnsgd.avg0995.lr1e_4",
	"dropout01.3l.n500.rmsprop.mom09.lr1e_3",
	"dropout01.3l.n500.smorms3.lr1e_3",
	"dropout01.3l.n500.smorms3.mom09.lr1e_3",
	"dropout01.3l.n500", # adadelta
	"dropout01.3l.n500.adadelta_decay090",
	"dropout01.3l.n500.adadelta_decay099",
	"dropout01.3l.n500.adadelta.lr01",
	"dropout01.3l.n500.adadelta.lr1e_2",
	"dropout01.3l.n500.adagrad.lr1e_2",
	"dropout01.3l.n500.adagrad.lr1e_3",
	"dropout01.3l.n500.adasecant.lr1",
	"dropout01.3l.n500.adasecant.lr05",
	"dropout01.3l.n500.adam.lr1e_2",
	"dropout01.3l.n500.adam.lr1e_3",
	"dropout01.3l.n500.adam.no_fit_lr.lr1e_3",
	"dropout01.3l.n500.nadam.lr1e_3",
	"dropout01.3l.n500.grad_noise03.adam.lr1e_3",
	"dropout01.3l.n500.upd_multi_model_2_2.adam.lr1e_3",
	"dropout01.3l.n500.upd_multi_model_3_2.adam.lr1e_3",
	"dropout01.3l.n500.adam.lr05e_3",
	"dropout01.3l.n500.adam.lr1e_4",
	"dropout01.3l.n500.adam.mnsgd.avg0995.lr1e_4",
	"dropout01.3l.n500.adam.lr1e_4.no_newbob",
	"dropout01.3l.n500.adam.lr1e_5.no_newbob",

	"dropout00.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout00.l2_1e_2.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_1.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.3l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_3.3l.n500.max_seqs40.adam.lr1e_3",

	"dropout01.l2_1e_2.2l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.3l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.4l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.5l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.6l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.7l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.8l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.9l.n500.max_seqs40.pretrain.adam.lr1e_3",
	"dropout01.l2_1e_2.10l.n500.max_seqs40.pretrain.adam.lr1e_3",

	"dropout01.l2_1e_2.1l.n500.max_seqs40.adam.lr1e_3",
	"dropout01.3l.n300.time_downsample2.max_seqs40.adam.lr1e_3",
	"dropout01.3l.n500.chunk100_75.max_seqs40.adam.lr1e_3",
	"dropout01.l2_1e_2.3l.n500.max_seqs40.adam.lr1e_3.upd_clip_01",
	"dropout01.l2_1e_2.5l.n500.max_seqs40.grad_noise03.adam.lr1e_3",
]

swb_dst_base_dir = mydir + "/switchboard"
swb_src_base_dir = "/u/zeyer/setups/switchboard/2016-01-28--crnn"
swb_experiments = [
	"dropout01.l2_1e_2.5l.n500.max_seqs40.grad_noise03.nadam.lr1e_3.grad_clip_inf",
	"vanilla_lstm.ontop.5l",
	"assoc_lstm.ontop.5l",
]


def cp(src_dir, dst_dir, filename):
	src_fn = src_dir + "/" + filename
	dst_fn = dst_dir + "/" + filename
	assert os.path.exists(src_fn), "%r does not exist" % src_fn
	try:
		os.makedirs(os.path.dirname(dst_fn))
	except os.error:
		pass
	print("copy (%s) %s" % (dst_dir, filename))
	shutil.copyfile(src_fn, dst_fn)


def main():
	for corpus_src, corpus_dst, experiments in [(quaero_src_base_dir, quaero_dst_base_dir, quaero_experiments), (swb_src_base_dir, swb_dst_base_dir, swb_experiments)]:
		for fn in base_files:
			cp(src_dir=corpus_src, dst_dir=corpus_dst, filename=fn)

		for setup_name in experiments:
			cp(
				src_dir=corpus_src,
				dst_dir=corpus_dst,
				filename="config-train/%s.config" % setup_name)
			cp(
				src_dir=corpus_src,
				dst_dir=corpus_dst,
				filename="scores/%s.recog.wers.txt" % setup_name)
			cp(
				src_dir=corpus_src,
				dst_dir=corpus_dst,
				filename="scores/%s.train.info.txt" % setup_name)


if __name__ == "__main__":
	main()
