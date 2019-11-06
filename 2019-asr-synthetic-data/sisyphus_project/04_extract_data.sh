#!/bin/bash

set -ex

cd data/dataset-raw

tar -xf dev-clean.tar.gz
tar -xf dev-other.tar.gz
tar -xf test-clean.tar.gz
tar -xf test-other.tar.gz
tar -xf train-clean-100.tar.gz
tar -xf train-clean-360.tar.gz
