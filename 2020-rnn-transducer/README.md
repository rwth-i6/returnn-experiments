RNN transducer (RNN-T), and variants.
[Paper link on Arxiv](https://arxiv.org/abs/2005.09319).

Paper title: A New Training Pipeline for an Improved Neural Transducer

Authors: Albert Zeyer, André Merboldt, Ralf Schlüter, Hermann Ney

The paper is a submission to Interspeech 2020.

Cite as:
```
@InProceedings { zeyer2020:transducer,
author= {Zeyer, Albert and Merboldt, André and Schlüter, Ralf and Ney, Hermann},
title= {A New Training Pipeline for an Improved Neural Transducer},
booktitle= {Interspeech},
year= 2020,
address= {http://www.interspeech2020.org/},
month= oct,
note= {to appear},
pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1145&row=pdf}
}
```

---

To reproduce our results, you would need
[RETURNN](https://github.com/rwth-i6/returnn),
and a copy of the Switchboard 300h audio corpus from LDC.
For the feature extraction,
we used [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/)
with [these RASR configs for Gammatone features (see `config` and `flow`)](https://github.com/rwth-i6/returnn-experiments/tree/master/2016-lstm-paper/switchboard).

The attention baseline is mostly based on [this](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-asr-e2e-trafo-vs-lstm/switchboard).

For BPE (1k), you can use a similar setup like
[here](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-asr-e2e-trafo-vs-lstm/tedlium2/full-setup)
or [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention).
The latter setup also includes the recognition setup,
which you can use as-is for all experiments (no matter if attention or transducer; even for CTC).

All the configs (always unified for training, recognition, alignment) are in [`configs`](configs).

You would first start the training of the setups needed to create alignments (e.g. the CTC model, or RNN-T/RNA full-sum).
Then you would use the script in [`dump-align`](dump-align)
to dump the (Viterbi) alignment to a file.

Then you can start the training of all remaining setups.

For recognition experiments on longer (concatenated) sequences,
you can use the script in [`concat-seqs`](concat-seqs) to prepare the recognition corpora.

For recognition experiments with varying beam sizes,
you would just pass another beam size to RETURNN
(or the [`search.py`](https://github.com/rwth-i6/returnn-experiments/blob/master/2018-asr-attention/librispeech/full-setup-attention/tools/search.py) script).

---

The best pipeline we came up with in the paper is:

* Full-sum training for 25 epochs, simple model, RNA label topology, [this config](https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config).
  (Note: In RETURNN, we split the epochs. In this config, into 6 parts. So RETURNN reports that it trains for 150 (sub) epochs. Which effectively means that it trains for 25 (real/full) epochs.)

* Extract alignments using the final full-sum trained model, using the script [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer/dump-align).

* Extended transducer model, start from scratch (no param import), train with frame-wise CE based on the extracted alignment, for 25 epochs, [this config](https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain.config).
* Train further 25 epochs with frame-wise CE, but reset learning rate, [this config](https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain.retrain1.config) (basically same config as before, but skips pretraining, and instead imports the parameter from the previous model).

So, you basically have 3 training stages here, but you throw away the first model after extracting the alignments.
(We are experimenting in variations of this training pipeline but have nothing to share at this moment.)

With respect to evaluating (any of the models at any stage), you can use the scripts/setup from [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention). See the recog there. (The setup is for Librispeech, so you need to be adapt this to the corpora.)
