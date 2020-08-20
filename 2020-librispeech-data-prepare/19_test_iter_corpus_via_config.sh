#!/bin/bash

set -exv

# Running this is not mandatory.
# Or you can also break it in the middle.
# This is just for testing, whether the dataset was prepared correctly.
# This will use the training config and iterate through epoch 1.
# For interactive terminals, you should see some progress bar like:
# 353/822, 0:00:04 [|||....]

# You might want to pass `--epoch 11` or so, to see the epoch-wise filter in effect.
# See `-h` for available options.

./returnn/tools/dump-dataset.py returnn.config --endseq -1 $*
