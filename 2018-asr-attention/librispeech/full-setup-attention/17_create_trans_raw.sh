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

