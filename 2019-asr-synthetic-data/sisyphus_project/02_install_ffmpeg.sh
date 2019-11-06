#!/bin/bash

set -ex

test -d src || mkdir src

cd src

wget https://ffmpeg.org/releases/ffmpeg-4.1.4.tar.gz
tar -xf ffmpeg-4.1.4.tar.gz

cd ffmpeg-4.1.4

./configure --disable-x86asm --enable-libvorbis
make -j 4
