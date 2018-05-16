#!/bin/bash

set -ex

mydir=$(pwd)
cd data/dataset

test -s stats.mean.txt && test -s stats.std_dev.txt && echo "stats.*.txt files already exist, exit" && exit

# 'seq_ordering':'random' just to have a better estimate of remaining time.
# Takes around 10h for me.
# bpe stuff not really needed here, just to make it load.

$mydir/returnn/tools/dump-dataset.py \
  "{'class':'LibriSpeechCorpus', 'bpe':{'bpe_file':'trans.bpe.codes', 'vocab_file':'trans.bpe.vocab'}, 'path':'.', 'audio':{}, 'prefix': 'train', 'use_zip': True, 'seq_ordering':'random'}" \
  --endseq -1 --type null --dump_stats stats

test -s stats.mean.txt && test -s stats.std_dev.txt
