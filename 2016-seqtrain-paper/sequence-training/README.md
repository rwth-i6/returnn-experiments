=== Faster Sequence Training
Authors: Albert Zeyer, Ilya Kulikov, Ralf Schlueter, Hermann Ney

This folder contains the configs which were used to perform experiments from the paper. They have to be used with RETURNN and RASR.
All experiments used data from CHiME-3 task, which is partially under the license, so we can distribute it. For further details 
about the data: http://spandh.dcs.shef.ac.uk/chime_challenge/chime2015/

`config-train` folder serves to provide configs for RETURNN.

`config` folder serves to provide configs for RASR.

`results` folder contains files with results obtained from performed experiments both training scores and recognitions wers.

To use the RETURNN configs with other data, replace the train/dev config settings, which specify the train and dev corpus data. At the moment, they will use the ExternSprintDataset interface to get the preprocessed data out of RASR. You can also use other dataset implementations provided by RETURNN (see RETURNN doc / source code), e.g. the HDF format directly.
