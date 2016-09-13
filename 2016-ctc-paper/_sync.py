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


swb_dst_base_dir = mydir + "/switchboard"
swb_src_base_dir = "/u/zeyer/setups/switchboard/2016-01-28--crnn"
swb_experiments = [
	"dropout01.l2_1e_2.5l.n500.max_seqs40.grad_noise03.nadam.lr1e_3.grad_clip_inf",  # framewise BLSTM baseline
	"ctcfbw.noff.tdpn.prior0",
	"ctcfbw.onlyff.tdpn.prior0",
	"ctcfbw.onlyff.tdpn.prior07.cpea0001",
	"ctcfbw.p2ff500b.tdpz",
]


def cp(src_dir, dst_dir, filename, optional=False):
	src_fn = src_dir + "/" + filename
	dst_fn = dst_dir + "/" + filename
	if not os.path.exists(src_fn):
		print("%r does not exist" % src_fn)
		assert optional
		return
	try:
		os.makedirs(os.path.dirname(dst_fn))
	except os.error:
		pass
	print("copy (%s) %s" % (dst_dir, filename))
	shutil.copyfile(src_fn, dst_fn)


def main():
	for corpus_src, corpus_dst, experiments in [(swb_src_base_dir, swb_dst_base_dir, swb_experiments)]:
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
				filename="scores/%s.recog.wers.txt" % setup_name,
				optional=True)
			cp(
				src_dir=corpus_src,
				dst_dir=corpus_dst,
				filename="scores/%s.train.info.txt" % setup_name)


if __name__ == "__main__":
	main()
