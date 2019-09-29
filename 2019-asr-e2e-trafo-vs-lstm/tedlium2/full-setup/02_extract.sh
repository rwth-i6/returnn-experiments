#!/bin/bash

set -ex

cd data

test -d dataset-extracted || mkdir dataset-extracted
cd dataset-extracted

tarf="../dataset-raw/TEDLIUM_release2.tar.gz"
test -e "$tarf"
tar xf "$tarf"

pwd
ls
