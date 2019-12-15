*the sisyphus setup is still under construction, but the RETURNN configurations for the ASR and TTS model can be found under sisyphus_project/config/returnn_configs/*


These configs are intended to help reproducing the results of the submitted ICASSP paper "GENERATING SYNTHETIC AUDIO DATA FOR ATTENTION-BASED SPEECH RECOGNITION SYSTEMS".
Although the full setup can not be published, this configs are intended to reproduce the most basics results on LibriSpeech-100h using TTS data for ASR.
External language models are not included.
The sisyphus setup is derived from the internal setup used to produce all presented results in the paper, and is still under construction.

#### Sisyphus Project

The folder sisyphus_project is a root directory for a sisyphus setup. The numbered shell scripts can be used to download the necessary tools, and prepare the data. If everything is set up, the sisyphus manager can be started to run all necessary jobs automatically

###### Installation

before starting the setup it is recommended to create a new python (minimum version 3.7) environment, to ensure all needed packages are correctly installed and there are no conflicts.
Then run all the shell scripts in their numbered order to install the software and download the data. Please be aware that the setup consumes a lot of hard drive space. At least 200 Gb of free space are recommended.





