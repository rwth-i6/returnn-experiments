#!/bin/bash

set -ex

test -d data
cd data

test -d dataset-extracted/TEDLIUM_release2

test -e dataset/train.zip
test -e dataset/dev.zip
test -e dataset/test.zip

test -d dataset-raw && rm -r dataset-raw

test -d dataset-extracted/TEDLIUM_release2/train && rm -r dataset-extracted/TEDLIUM_release2/train
test -d dataset-extracted/TEDLIUM_release2/dev && rm -r dataset-extracted/TEDLIUM_release2/dev
test -d dataset-extracted/TEDLIUM_release2/test && rm -r dataset-extracted/TEDLIUM_release2/test

test -e dataset/train.txt && rm dataset/train.txt
test -d dataset/train && rm -r dataset/train
test -e dataset/dev.txt && rm dataset/dev.txt
test -d dataset/dev && rm -r dataset/dev
test -e dataset/test.txt && rm dataset/test.txt
test -d dataset/test && rm -r dataset/test
