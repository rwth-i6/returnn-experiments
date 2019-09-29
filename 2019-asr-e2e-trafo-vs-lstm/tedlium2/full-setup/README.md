TEDLIUM corpus version2

A. Rousseau, P. Deléglise, and Y. Estève, "Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks", in Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), May 2014.

~200h TED talks.

This is a complete setup pipeline to end up with an encoder-attention-decoder system
on the speech recognition task [TED-LIUM-v2 200h](https://lium.univ-lemans.fr/en/ted-lium2/)
using [RETURNN](https://github.com/rwth-i6/returnn),
as it is described in our paper
[A comparison of Transformer and LSTM encoder decoder models for ASR](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-asr-e2e-trafo-vs-lstm),
which yields very competitive results.

```
@InProceedings { zeyer2019:trafo-vs-lstm-asr,
author= {Zeyer, Albert and Bahar, Parnia and Irie, Kazuki and Schlüter, Ralf and Ney, Hermann},
title= {A comparison of Transformer and LSTM encoder decoder models for ASR},
booktitle= {IEEE Automatic Speech Recognition and Understanding Workshop},
year= 2019,
address= {Sentosa, Singapore},
month= dec,
booktitlelink= {http://asru2019.org/wp/}
}
```

This setup also includes the dataset download and preparation.
(This is a similar script as for Librispeech [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention).)

Required tools for the scripts:

* bash, python3
* tar, zip
* ffmpeg (SPH files, Ogg files)

Required Python packages (via `pip3 install ...`), see requirementx.txt.

All the data will be created in a `data` subdirectory.
The raw dataset files will be downloaded to `data/dataset-raw`.
This can be deleted once the dataset preparation is finished.
All the prepared dataset files will be in `data/dataset`.

Note that the dataset should be on a fast file system. NFS will make the training much slower!
(At our chair, we use NFS together with the [CacheManager software](https://github.com/pavelgolik/cache-manager).
 RETURNN has the option `use_cache_manager` in the `OggZipDataset` to use that.)

The recognition will automatically select a few interesting epochs (via cross validation scores).
Then it will select the best model by the best WER from the dev-other dataset.
