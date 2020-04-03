
BPE-level Transformer language model training configs for TED-LIUM 2 (ASRU 2019).

* Training using all TED-LIUM 2 language model training data: `all.nopos_transfo_30_d0.4096_768.8h.lr1.cl1.config`
* Fine-tuning on the most relevant subsets (transcriptions and commoncrawl): `fine_cc_train.all.nopos_transfo_30_d0.4096_768.8h.lr1.cl1.config`

NB: For word-level language models, and further discussion on language modeling data of TED-LIUM 2, we have a more recent paper (ICASSP 2020).
The corresponding repository is:

https://github.com/rwth-i6/returnn-experiments/tree/master/2020-lm-small-state-trafo
