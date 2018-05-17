#!/bin/bash

set -exv

test -e data/exp-returnn/train-scores.data

epochs=$(./tools/recommend-recog-epochs.py)

for epoch in $epochs; do
  echo "recog of epoch $epoch"
  ./tools/search.py returnn $epoch --data dev-clean
  ./tools/search.py returnn $epoch --data dev-other
  ./tools/search.py returnn $epoch --data test-clean
  ./tools/search.py returnn $epoch --data test-other
done
