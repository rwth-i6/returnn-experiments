#!/bin/bash

set -ex

test -d data/dataset
cd data/dataset

function create_zip {
  local name=$1  # "train", "dev" or "test"
  local dest=$name.zip
  echo "create $dest"
  test -e $name.txt
  test -d $name
  zip -r $dest $name.txt $name
}

create_zip "train"
create_zip "dev"
create_zip "test"

unzip -l train.zip
unzip -l dev.zip
unzip -l test.zip

