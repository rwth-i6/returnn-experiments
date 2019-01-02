This language model (LM) be used together with the [attention model](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention).
See [here](https://github.com/rwth-i6/returnn-experiments/blob/master/2018-asr-attention/librispeech/attention/exp3.ctc.lm.config) for an example.

A pretrained model can be downloaded [here](http://www-i6.informatik.rwth-aachen.de/~zeyer/models/librispeech/lm/bpe-10k/2018.irie.i512_m2048_m2048.sgd_b64_lr0_cl2.newbobabs.d0.2/).

The vocab used for the `LmDataset` has a custom format, different from the attention model
(which you need, if you want to train it yourself).
It should be straight forward to convert from one to the other.
Or to add support for the other format in `LmDataset`. 
The LM vocab file can be downloaded [here](http://www-i6.informatik.rwth-aachen.de/~zeyer/models/librispeech/lm/bpe-10k/trans.bpe.vocab.lm.txt).
The train files (data_files in config) are generated from the [LibriSpeech LM training data](http://www.openslr.org/11/).
