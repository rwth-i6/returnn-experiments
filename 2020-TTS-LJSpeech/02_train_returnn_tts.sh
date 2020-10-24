#!/usr/bin/env bash
export OMP_NUM_THREADS=4
export NUMBA_CACHE_DIR=/var/tmp
python3 returnn/rnn.py tacotron2_ljspeech.config
