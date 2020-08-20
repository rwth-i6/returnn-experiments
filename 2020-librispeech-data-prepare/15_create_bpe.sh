#!/bin/bash

set -ex
mydir=$(pwd)
cd data/dataset

test -s train-trans-all.txt || { $mydir/tools/collect-train-text.py > train-trans-all.txt; }
wc -l train-trans-all.txt
test $(wc -l train-trans-all.txt | awk {'print $1'}) -eq 281241

test -s trans.bpe.codes || $mydir/subword-nmt/learn_bpe.py --input train-trans-all.txt --output trans.bpe.codes --symbols 10000

test -s trans.bpe.vocab || \
	$mydir/subword-nmt/create-py-vocab.py --txt train-trans-all.txt --bpe trans.bpe.codes --unk "<unk>" \
	--out trans.bpe.vocab

