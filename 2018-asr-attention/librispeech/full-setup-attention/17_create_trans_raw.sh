#!/bin/bash

set -exv

# This is just needed to calculate the final word-error-rate (WER).

for prefix in dev-clean dev-other test-clean test-other train-clean train-other; do
    if test -s data/dataset/$prefix.trans.raw; then
      echo "$prefix exists already"
      continue
    fi

    ./returnn/tools/dump-dataset-raw-strings.py --dataset "{'class':'LibriSpeechCorpus', 'bpe':{'bpe_file':'data/dataset/trans.bpe.codes', 'vocab_file':'data/dataset/trans.bpe.vocab'}, 'path':'data/dataset', 'audio':{}, 'prefix': '$prefix', 'use_zip': True}" --out data/dataset/$prefix.trans.raw

done

wc -l data/dataset/*.trans.raw

test $(wc -l data/dataset/dev-clean.trans.raw | awk {'print $1'}) -eq 2705
test $(wc -l data/dataset/dev-other.trans.raw | awk {'print $1'}) -eq 2866
test $(wc -l data/dataset/test-clean.trans.raw | awk {'print $1'}) -eq 2622
test $(wc -l data/dataset/test-other.trans.raw | awk {'print $1'}) -eq 2941
test $(wc -l data/dataset/train-clean.trans.raw | awk {'print $1'}) -eq 132555
test $(wc -l data/dataset/train-other.trans.raw | awk {'print $1'}) -eq 148690
