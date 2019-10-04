These are configs for encoder-decoder-attention models.

* As an update, we added a more recent config, `base2.conv2l.specaug.curric3`,
  which makes use of [SpecAugment](https://arxiv.org/abs/1904.08779).
  This config gets 10.7% WER on dev-other in epoch 250 (without LM).
  See [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-asr-e2e-trafo-vs-lstm/librispeech)
  for more recent configs, and [this corresponding paper](https://www-i6.informatik.rwth-aachen.de/publications/download/1119/Zeyer-ASRU-2019.pdf).

* The best config from the paper (without LM) is `base3.retrain2`, epoch 166,
  with 12.93% WER on dev-other.
  This is a continued training from the best model of `base3.retrain`
  (13.07% WER on dev-other, epoch 132),
  which itself is a continued training from the best model of `base2.bs18k.curric3`
  (13.32% WER, epoch 250).

* `base2.bs18k.curric3` is based on `base2.bs18k` with the modified curriculum learning,
  with 13.32% WER on dev-other in epoch 250.

* `base2.bs18k` is based on `base2` with batch size 18000 instead of 20000.

* `base2` is basically the same as `base.red6.pretrain-start2l-red6-below-grow3`.

* `base.red6.pretrain-start2l-red6-below-grow3` is the best LibriSpeech result as reported
  in the [A comprehensive analysis on attention models](https://www-i6.informatik.rwth-aachen.de/publications/download/1091/Zeyer-NIPS%20IRASL-2018.pdf) 2018 NIPS IRASL paper,
  with 13.95% WER on dev-other in epoch 239.
  Related files can also be found [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-nips-irasl-paper/librispeech).
  
* Older setups from the [Improved training of end-to-end attention models for speech recognition](https://www-i6.informatik.rwth-aachen.de/publications/download/1068/Zeyer--2018.pdf) 2018 Interspeech paper
  can be found [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/attention).
  A full training pipeline including preprocessing is [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention)
  which can be adopted with these new configs.
