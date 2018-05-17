#!/bin/bash

set -exv

# Running this is not mandatory.
# Or you can also break it in the middle.
# This is just for testing, whether the dataset was prepared correctly.
# For interactive terminals, you should see some progress bar like:
# 209/281241, 3:05:31 [ 0.07% ]

./returnn/tools/dump-dataset.py \
  "{'class':'LibriSpeechCorpus', 'bpe':{'bpe_file':'data/dataset/trans.bpe.codes', 'vocab_file':'data/dataset/trans.bpe.vocab'}, 'path':'data/dataset', 'audio':{'norm_mean':'data/dataset/stats.mean.txt', 'norm_std_dev':'data/dataset/stats.std_dev.txt'}, 'prefix': 'train', 'use_zip': True}" \
  --endseq -1 --type null

