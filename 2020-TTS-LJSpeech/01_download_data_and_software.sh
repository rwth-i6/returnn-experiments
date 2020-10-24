#!/bin/bash

# download PWG and RETURNN from git
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
git clone https://github.com/rwth-i6/returnn.git

# create data folder and download training data
mkdir -P data
cd data

wget http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/LJSpeech.ogg.zip 
wget http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/cmudict.dict 
wget http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/cmu_vocab.pkl 
wget http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/dev_segments.txt 
wget http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/train_segments.txt 
