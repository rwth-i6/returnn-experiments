This is a complete setup pipeline to end up with an encoder-attention-decoder system on LibriSpeech 1000h,
as it is described in our paper [Improved training of end-to-end attention models for speech recognition](https://arxiv.org/abs/1805.03294).
It also includes the dataset download and preparation.

All the data will be created in a `data` subdirectory.
The raw dataset files will be downloaded to `data/dataset-raw` (about 58GB). This can be deleted once the dataset preparation is finished.
All the prepared dataset files will be in `data/dataset` (about 60GB).

Note that the dataset should be on a fast file system. NFS will make the training much slower!
(At our chair, we use NFS together with the [CacheManager software](https://www-i6.informatik.rwth-aachen.de/~rybach/cache-manager.php).
 RETURNN has the option `use_cache_manager` in the `LibriSpeechCorpus` to use that.)

The RETURNN training/model config `returnn.config` is based on `../attention/exp3.ctc.config`
and adapted for the data pathes here.
See `../attention/scores` for reference what train/dev scores to expect,
and also what recognition word-error-rate (WER) to expect.
