#!/bin/bash

set -ex

# Convert tar.gz files to zip files, including ogg files.
# This is what we use in RETURNN dataset, because
# it allows better random access inside the file.

cd data
zipdir="$(pwd)"/dataset-ogg
test -d $zipdir
test -e $zipdir/train-other-500.zip

tardir="$(pwd)"/dataset-raw
test -d $tardir
rm -r $tardir
