#!/bin/bash

set -ex

cd data  # make sure that directory exists, and can store lots of data (~100GB temporarily)
test -d dataset-raw || mkdir dataset-raw
cd dataset-raw

wget -c "https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release2.tar.gz"
