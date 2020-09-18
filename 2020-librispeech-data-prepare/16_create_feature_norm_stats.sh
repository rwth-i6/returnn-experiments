#!/bin/bash

set -ex

mydir=$(pwd)
cd data/dataset

test -s stats.mean.txt && test -s stats.std_dev.txt && echo "stats.*.txt files already exist, exit" && exit

# 'seq_ordering':'random' just to have a better estimate of remaining time.
# Takes around 2h for me.
# bpe stuff not really needed here, just to make it load.

audio="{}"  # default settings, i.e. MFCC

files=""
for part in train-clean-100 train-clean-360 train-other-500; do
  test -e ../dataset-ogg/$part.zip
  test -e ../dataset-ogg/$part.txt.gz
  files="$files'../dataset-ogg/$part.zip', '../dataset-ogg/$part.txt.gz', "
done

$mydir/returnn/tools/dump-dataset.py \
  "{'class':'OggZipDataset', 'targets':None, 'path':[$files], 'audio':$audio, 'zip_audio_files_have_name_as_prefix':False, 'seq_ordering':'random'}" \
  --endseq -1 --type null --dump_stats stats

test -s stats.mean.txt && test -s stats.std_dev.txt
