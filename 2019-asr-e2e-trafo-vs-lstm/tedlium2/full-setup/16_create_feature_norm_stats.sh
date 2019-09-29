#!/bin/bash

set -ex

mydir=$(pwd)
cd data/dataset

test -s stats.mean.txt && test -s stats.std_dev.txt && echo "stats.*.txt files already exist, exit" && exit

# 'seq_ordering':'random' just to have a better estimate of remaining time.
# Takes around 40 min for me.
# bpe stuff not really needed here, just to make it load.

$mydir/returnn/tools/dump-dataset.py \
  "{'class':'OggZipDataset', 'targets':{'class':'BytePairEncoding', 'bpe_file':'trans.bpe1k.codes', 'vocab_file':'trans.bpe1k.vocab', 'unknown_label':'<unk>'}, 'path':'train.zip', 'audio':{}, 'seq_ordering':'random'}" \
  --endseq -1 --type null --dump_stats stats

test -s stats.mean.txt && test -s stats.std_dev.txt
