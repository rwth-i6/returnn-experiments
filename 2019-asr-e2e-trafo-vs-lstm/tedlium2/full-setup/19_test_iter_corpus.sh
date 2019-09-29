#!/bin/bash

set -exv

# Running this is not mandatory.
# Or you can also break it in the middle.
# This is just for testing, whether the dataset was prepared correctly.
# For interactive terminals, you should see some progress bar like:
# 258/92973, 0:57:07 [ 0.28% ]

./returnn/tools/dump-dataset.py \
  "{'class':'OggZipDataset', 'targets':{'class':'BytePairEncoding', 'bpe_file':'data/dataset/trans.bpe1k.codes', 'vocab_file':'data/dataset/trans.bpe1k.vocab', 'unknown_label':'<unk>'}, 'path':'data/dataset/train.zip', 'audio':{'norm_mean':'data/dataset/stats.mean.txt', 'norm_std_dev':'data/dataset/stats.std_dev.txt'}}" \
  --endseq -1 --type null

