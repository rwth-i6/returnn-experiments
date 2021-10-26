#!/usr/bin/bash
PY=python3

for align in $(ls | grep -v vocab | grep -v map | grep -v '.sh')
do
  $py ../../scripts/parse_g2p-align.py $align $align.vocab $align.map
done
