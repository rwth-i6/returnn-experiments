#!/bin/bash

set -exv

# Running this is not mandatory.
# Or you can also break it in the middle.
# This is just for testing, whether the dataset was prepared correctly.
# For interactive terminals, you should see some progress bar like:
# 209/281241, 3:05:31 [ 0.07% ]

targets="{'bpe_file':'data/dataset/trans.bpe.codes', 'vocab_file':'data/dataset/trans.bpe.vocab', 'unknown_label':'<unk>'}"
audio="{'norm_mean':'data/dataset/stats.mean.txt', 'norm_std_dev':'data/dataset/stats.std_dev.txt'}"

files=""
for part in train-clean-100 train-clean-360 train-other-500; do
  test -e ../dataset-ogg/$part.zip
  test -e ../dataset-ogg/$part.txt.gz
  files="$files'../dataset-ogg/$part.zip', '../dataset-ogg/$part.txt.gz', "
done

./returnn/tools/dump-dataset.py \
  "{'class':'OggZipDataset', 'path':[$files], 'audio':$audio, 'targets':$targets, 'zip_audio_files_have_name_as_prefix':False}" \
  --endseq -1 --type null
