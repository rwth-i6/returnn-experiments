We propose new methods to estimate the internal language model for attention-based encoder-decoder ASR models.

Paper: [Investigating Methods to Improve Language Model Integration for Attention-based Encoder-Decoder ASR Models](https://arxiv.org/abs/2104.05544)

Please cite as:

```
@misc{zeineldeen2021investigating,
      title={Investigating Methods to Improve Language Model Integration for Attention-based Encoder-Decoder ASR Models}, 
      author={Mohammad Zeineldeen and Aleksandr Glushko and Wilfried Michel and Albert Zeyer and Ralf Schlüter and Hermann Ney},
      year={2021},
      eprint={2104.05544},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Features

For Switchboard, we use 40-dim Gammatones features extracted using [RASR](https://github.com/rwth-i6/rasr). For Librispeech, we use 40-dim MFCC features extracted using `librosa` within [RETURNN](https://github.com/rwth-i6/returnn).

### Configs

All training and inference are done using [RETURNN](https://github.com/rwth-i6/returnn). All configs are generated using [Sisyphus](https://github.com/rwth-i6/sisyphus).

#### ASR

- `ffdec_*`: configs for Feed-forward (FF) decoder models
- `lstmdec_*`: configs for LSTM-based decoder models

#### Language Model (LM)

Configs for training transcription only LMs for Density Ratio and for training external LMs are found under directory `lm`.

#### Internal Language Model (ILM)

- `train_mini_lstm.config`: config to train Mini-LSTM

##### Inference configs with ILM estimation methods


- `ilm_zero.config`: uses zero method
- `ilm_global_avg.config`: uses global average (encoder or attention) method
- `ilm_seq_avg.config`: uses sequence averge method
- `ilm_mini_lstm.config`: uses Mini-LSTM method

For dumping the global average over training data for encoder outputs or attention context vectors, we used this [script](https://github.com/rwth-i6/returnn/blob/master/tools/dump-forward-stats.py).
