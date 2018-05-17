#!/bin/bash

set -exv

epochs=$(./tools/recommend-recog-epochs.py)

for epoch in $epochs; do
  echo "recog of epoch $epoch"
  ./tools/search.py returnn $epoch
done
