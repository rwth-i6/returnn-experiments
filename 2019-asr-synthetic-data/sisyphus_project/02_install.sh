#!/bin/bash

set -ex

test -d src || mkdir src

# This is for creating the BPE vocab.
git clone https://github.com/albertz/subword-nmt.git src/subword-nmt

# RETURNN
git clone https://github.com/rwth-i6/returnn.git src/returnn

# Sisyphus
git clone https://github.com/rwth-i6/sisyphus.git src/sisyphus

# enables autocompletion for sisyphus when opening the experiment folder with PyCharm
ln -s src/sisyphus/sisyphus sisyphus