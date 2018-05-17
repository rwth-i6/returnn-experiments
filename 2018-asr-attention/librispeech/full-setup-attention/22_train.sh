#!/bin/bash

set -exv

# If you stop and rerun this, it will continue from the last epoch.
# All data (models, logs and train/dev scores) will be in data/exp-returnn.
# train/dev scores in particular are in the file data/exp-returnn/train-scores.data.

./returnn/rnn.py returnn.config
