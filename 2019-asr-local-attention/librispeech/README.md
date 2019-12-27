`*.config` files are [RETURNN config files](https://github.com/rwth-i6/returnn),
the training/recog files are in `scores/`.

The following table documents which files correspond to which model:

| Model in paper  | Modelname |
| ------------- | ------------- |
| baseline (global)  | `exp3.ctc`  |
| local (argmax)  | `local-heuristic.argmax.win{02,05,08,10,12,15,20}.exp3.ctc`  |

Unfortunately the configs are slightly buggy, and do not work correctly with a more recent RETURNN version (2019-12-27).
It should be easy to fix, though. Change this:

    "p_t_in": {"class": "eval", "from": "prev:att_weights", "eval": "tf.squeeze(tf.argmax(source(0), axis=1, output_type=tf.int32), axis=1)",
      "out_type": {"shape": (), "batch_dim_axis": 0, "dtype": "float32"}},

To:

    "p_t_in": {"class": "reduce", "from": "prev:att_weights", "mode": "argmax", "axis": "t"},

---

Also see the configs from our 2019 LibriSpeech system paper from the same year
[here](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-librispeech-system/attention).

An older full training pipeline can be found [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention),
which can be adopted for these new configs.
