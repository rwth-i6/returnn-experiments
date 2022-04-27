This repository contains configs needed to reproduce our results for our submitted paper to INTERSPEECH 2022.

Paper title: **Improving the Training Recipe for a Robust Conformer-based Hybrid Model**

### Features Extraction

We use 40-dim Gammatones features extracted using [RASR](https://github.com/rwth-i6/rasr).

### Models Configs

[REURNN](https://github.com/rwth-i6/returnn) is used to train all models. We use [RASR](https://github.com/rwth-i6/rasr) for recognition.

#### For training

- `tabX_*`: directory containing the configs for table X in the paper

#### For recognition

RASR recognition configs can be found in `recognition` directory.


