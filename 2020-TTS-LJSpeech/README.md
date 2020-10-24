RETURNN TTS LJSpeech minimal setup
##################################

This setup is a demo for the Interspeech 2020 tutorial to show a RETURNN TTS setup in combination with an external PyTorch vocoder.

Installation
------------

The listed packages can be found in the `requirements.txt`.
You need to have a valid CUDA installation with preferably CUDA 10.1 (other versions might work as well, please adapt the torch requirement then).
Calling `01_download_data_and_software.sh` will download the required data, as well as the RETURNN and ParallelWaveGAN repository. 

Pre-trained models can be found under `http://www-i6.informatik.rwth-aachen.de/~rossenbach/models/LJSpeech_TTS/data/pretrained`

Training
--------

Follow the scripts 1 to 5 to run the TTS feature model training and the vocoder training. The current settings require a GPU with a memory of 10 Gb to run. If you encounter out-of-memory issues, reduce the batch size in the config files.

Decoding
--------

For decoding, pipe any text file into the `06_decode.sh`. The resulting audio will be stored as `out_<line-nr>.wav`. Note that words that are not part of the CMU pronuncation dictionary will be dropped. 

