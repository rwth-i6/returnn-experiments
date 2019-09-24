This repo contains the configs and related files
to be used with [RETURNN](https://github.com/rwth-i6/returnn)
(called CRNN earlier/internally)
and [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/)
(called Sprint internally)
for data preprocessing and decoding.

To use the RETURNN configs with other data,
replace the `train`/`dev` config settings, which specify the train and dev corpus data.
At the moment, they will use the `ExternSprintDataset` interface to get the preprocessed data out of RASR.
You can also use other dataset implementations provided by RETURNN (see RETURNN doc / source code),
e.g. the HDF format directly.

Most of these configs belong to
[publications from our group](https://www-i6.informatik.rwth-aachen.de/web/Publications/index.html),
[Human Language Technology and Pattern Recognition Group (i6), RWTH Aachen University, Germany](https://www-i6.informatik.rwth-aachen.de).
