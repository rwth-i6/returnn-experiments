This repository contains configs needed to reproduce our results for our accepted paper at INTERSPEECH 2022.

Paper: [Improving the Training Recipe for a Robust Conformer-based Hybrid Model](https://arxiv.org/abs/2206.12955)

### Features Extraction

We use 40-dim Gammatones features extracted using [RASR](https://github.com/rwth-i6/rasr).

### Experiments Configs

- [REURNN](https://github.com/rwth-i6/returnn) is used to train all models. We use [RASR](https://github.com/rwth-i6/rasr) for recognition.

- `table_X`: directory containing the configs for table(s) X in the paper

