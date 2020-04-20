Paper: [Generating Synthetic Audio Data for Attention-Based Speech Recognition Systems](https://www-i6.informatik.rwth-aachen.de/publications/download/1128/Rossenbach-ICASSP-2020.pdf)

Please cite as:
```
@InProceedings { rossenbach:ICASSP2020,
author= {Rossenbach, Nick and Zeyer, Albert and Schlüter, Ralf and Ney, Hermann},
title= {Generating Synthetic Audio Data for Attention-Based Speech Recognition Systems},
booktitle= {IEEE International Conference on Acoustics, Speech, and Signal Processing},
year= 2020,
address= {Barcelona, Spain},
month= may,
booktitlelink= {https://2020.ieeeicassp.org/},
pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1128&row=pdf}
}
```

*the sisyphus setup is still under construction, but the RETURNN configurations for the ASR and TTS model can be found under sisyphus_project/config/returnn_configs/*


These configs are intended to help reproducing the results of the submitted ICASSP paper "GENERATING SYNTHETIC AUDIO DATA FOR ATTENTION-BASED SPEECH RECOGNITION SYSTEMS".
Although the full setup can not be published, this configs are intended to reproduce the most basics results on LibriSpeech-100h using TTS data for ASR.
External language models are not included.
The sisyphus setup is derived from the internal setup used to produce all presented results in the paper, and is still under construction.

The models are trained with [**RETURNN**](https://www.github.com/rwth-i6/returnn) and the experiments are designed 
with [**Sisyphus**](https://www.github.com/rwth-i6/sisyphus). The "raw" RETURNN configs for the ASR and TTS models can
 be found in 
`sisyphus_project/config/returnn-configs/`


### Sisyphus Project

The folder sisyphus_project is a root directory for a sisyphus setup. The numbered shell scripts can be used to download the necessary tools, and prepare the data. If everything is set up, the sisyphus manager can be started to run all necessary jobs automatically

#### Installation

before starting the setup it is recommended to create a new python (minimum version 3.7) environment, to ensure all needed packages are correctly installed and there are no conflicts.
Then run all the shell scripts in their numbered order to install the software and download the data. Please be aware that the setup consumes a lot of hard drive space. At least 200 Gb of free space are recommended. Be sure to `source` the correct `python3` environment before launching the install scripts.


_1: Install the required packages with pip:_
```
./01_pip_install_requirements.sh
```

_2: Clone the required software (RETURNN, Sisyphus, subword-nmt) and download and compile ffmpeg_
```
./02_install.sh
```
Although preinstalled ffmpeg versions can be used, this might change the audio processing due to version differences.

_3. Download the LibriSpeech dataset_
```
./03_download_data.sh
```
Per default the script downloads only LibriSpeech-100h and LibriSpeech-360h, if you want to use also LibriSpeech-500h, uncommend the according line.

_4. Extract the downloaded data:_
```
./04_extract_data.sh
```
Again, uncommend the line for LibriSpeech-500h if necessary.

#### Running Sisyphus

The execution can be started by calling:
```
./sis.sh m
```
`m` is a shortcut for `mananger` which is the main module of sisyphus organizing and launching the jobs. When 
providing `-r`, the execution will directly start without a prompt.


