#!/bin/bash

set -ex

cd data
test -d dataset-raw || mkdir dataset-raw
cd dataset-raw

# via: http://www.openslr.org/12/
# download. or continue download. or skip if existing and completed.
wget -c http://www.openslr.org/resources/12/dev-clean.tar.gz
wget -c http://www.openslr.org/resources/12/dev-other.tar.gz
wget -c http://www.openslr.org/resources/12/test-clean.tar.gz
wget -c http://www.openslr.org/resources/12/test-other.tar.gz
wget -c http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c http://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c http://www.openslr.org/resources/12/train-other-500.tar.gz
wget -c http://www.openslr.org/resources/12/md5sum.txt

md5sum -c md5sum.txt
