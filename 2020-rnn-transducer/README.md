RNN transducer (RNN-T), and variants.
[Paper link on Arxiv](https://arxiv.org/abs/2005.09319).

Paper title: A New Training Pipeline for an Improved Neural Transducer

Authors: Albert Zeyer, André Merboldt, Ralf Schlüter, Hermann Ney

The paper is a submission to Interspeech 2020.

Cite as:
```
@misc{zeyer2020transducer,
    title={A New Training Pipeline for an Improved Neural Transducer},
    author={Albert Zeyer and André Merboldt and Ralf Schlüter and Hermann Ney},
    year={2020},
    howpublished={Preprint arXiv:2005.09319}
}
```
(This will be updated if the paper gets accepted for Interspeech.)

---

To reproduce our results, you would need
[RETURNN](https://github.com/rwth-i6/returnn),
and a copy of the Switchboard 300h audio corpus from LDC.
For the feature extraction,
we used [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/)
with [these configs for Gammatone features](https://github.com/rwth-i6/returnn-experiments/tree/master/2016-lstm-paper/switchboard).

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
