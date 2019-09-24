`*.config` files are [RETURNN config files](https://github.com/rwth-i6/returnn),
the training/recog files are in `scores/`.

The following table documents which files correspond to which model:

| Model in paper  | Modelname |
| ------------- | ------------- |
| baseline (global)  | `exp3.ctc`  |
| local (argmax)  | `local-heuristic.argmax.win{02,05,08,10,12,15,20}.exp3.ctc`  |

---

Also see the configs from our 2019 LibriSpeech system paper from the same year
[here](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-librispeech-system/attention).

An older full training pipeline can be found [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention),
which can be adopted for these new configs.
