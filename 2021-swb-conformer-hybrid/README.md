We present a training recipe for conformer-based hybrid ASR models.

### Features Extraction

We use 40-dim Gammatones features extracted using [RASR](https://github.com/rwth-i6/rasr).

### Models Configs

[REURNN](https://github.com/rwth-i6/returnn) is used to train all models. We use [RASR](https://github.com/rwth-i6/rasr) for recognition.

- `baseline_12l.config`: baseline config
- `tabX_*`: directory containing the configs for table X in the paper
- `smbr_train.config`: sMBR training config
