#!/bin/bash

set -exv

experiment=returnn
test -e data/exp-$experiment  # experiment existing?
test -e data/exp-$experiment/train-scores.data  # some epochs trained?

epochs=$(./tools/recommend-recog-epochs.py --experiment $experiment)
extra_args="-- ++batch_size 2000"  # such that it fits on your GPU

for epoch in $epochs; do
  echo "recog of epoch $epoch"
  ./tools/search.py $experiment $epoch --data dev-clean $extra_args
  ./tools/search.py $experiment $epoch --data dev-other $extra_args
  ./tools/search.py $experiment $epoch --data test-clean $extra_args
  ./tools/search.py $experiment $epoch --data test-other $extra_args
done
