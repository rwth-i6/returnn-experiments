#!/bin/bash

set -exv

# This is just needed to calculate the final word-error-rate (WER).

mydir=$(pwd)
cd data/dataset
test -e train.zip

for name in train dev test; do
    dest=$name.trans.raw
    if test -s $dest; then
      echo "$dest exists already"
      continue
    fi

    test -e $name.zip
    $mydir/returnn/tools/dump-dataset-raw-strings.py --dataset "{'class':'OggZipDataset', 'targets':{'bpe_file':'trans.bpe1k.codes', 'vocab_file':'trans.bpe1k.vocab', 'unknown_label':'<unk>'}, 'path':'$name.zip', 'audio':None}" --out $dest

done

wc -l *.trans.raw

test $(wc -l train.trans.raw | awk {'print $1'}) -eq 92975
test $(wc -l dev.trans.raw | awk {'print $1'}) -eq 509
test $(wc -l test.trans.raw | awk {'print $1'}) -eq 1159
