#!/bin/bash
PYTHONPATH=ParallelWaveGAN python3 ParallelWaveGAN/parallel_wavegan/bin/train.py --config mb_melgan.v2.yaml --train-dumpdir data/gta_dataset/LJSpeech --dev-dumpdir data/gta_dataset/LJSpeech_dev/ --outdir mb_melgan_models/ --verbose 2
