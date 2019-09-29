#!/bin/bash

set -ex
mydir=$(pwd)
cd data/dataset
test -e train.zip

N=5000
bpe_file=trans.bpe5k

test -s train-trans.txt || { $mydir/tools/collect-dataset-text.py train.zip > train-trans.txt; }
wc -l train-trans.txt
test $(wc -l train-trans.txt | awk {'print $1'}) -eq 92973  # check that number of seqs is correct

test -s $bpe_file.codes || $mydir/subword-nmt/learn_bpe.py --input train-trans.txt --output $bpe_file.codes --symbols $N

test -s $bpe_file.vocab || \
    $mydir/subword-nmt/create-py-vocab.py --txt train-trans.txt --bpe $bpe_file.codes --unk "<unk>" \
    --out $bpe_file.vocab
