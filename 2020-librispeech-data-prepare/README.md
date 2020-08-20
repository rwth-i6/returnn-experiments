This is a the pipeline to download and prepare the data
of the speech recognition task [LibriSpeech 1000h](http://www.openslr.org/12/)
to be used with [RETURNN](https://github.com/rwth-i6/returnn).
This is based on the earlier pipeline from [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention),
with some components from [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-asr-e2e-trafo-vs-lstm/tedlium2/full-setup).

All the data will be created in a `data` subdirectory.
The raw dataset files will be downloaded to `data/dataset-raw` (about 58GB).
This can be deleted once the dataset preparation is finished
(e.g. via `13_cleanup_data.sh`).
All the prepared dataset files will be in `data/dataset` and `data/dataset-ogg` (about 16GB).

Note that the dataset should be on a fast file system. NFS will make the training much slower!
(At our chair, we use NFS together with the [CacheManager software](https://github.com/pavelgolik/cache-manager).
 RETURNN has the option `use_cache_manager` in the `LibriSpeechCorpus` to use that.)
