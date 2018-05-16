#!/bin/bash

set -ex

mydir=$(pwd)
cd data/dataset

test -s stats.mean.txt && test -s stats.std_dev.txt && echo "stats.*.txt files already exist, exit" && exit

$mydir/returnn/tools/dump-dataset.py \
  "{'class':'LibriSpeechCorpus', 'bpe':{'bpe_file':'trans.bpe.codes', 'vocab_file':'trans.bpe.vocab'}, 'path':'.', 'audio':{}, 'prefix': 'train', 'use_zip': True}" \
  --endseq -1 --type null --dump_stats stats

test -s stats.mean.txt && test -s stats.std_dev.txt
