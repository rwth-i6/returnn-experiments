#!/bin/bash

set -ex

./collect-text.py > train-trans-all.txt

./subword-nmt/learn_bpe.py --input train-trans-all.txt --output trans.bpe.codes --symbols 10000

./subword-nmt/create-py-vocab.py --txt train-trans-all.txt --bpe trans.bpe.codes --out trans.bpe.vocab

