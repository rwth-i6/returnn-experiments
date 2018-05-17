#!/bin/bash

set -exv

# This will create the data dir right here.
# You might not want this.
# Create it on a fast file system, where you have plenty of space (>200GB),
# and then symlink data instead.
# See also README.

test -d data || mkdir data
