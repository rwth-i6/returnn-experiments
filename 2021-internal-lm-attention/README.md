We propose new methods to estimate the internal language model for attention-based encoder-decoder ASR models.

Paper: [Investigating Methods to Improve Language Model Integration for Attention-based Encoder-Decoder ASR Models](https://arxiv.org/abs/2104.05544)

Please cite as:

```
@inproceedings{DBLP:conf/interspeech/ZeineldeenGMZSN21,
  author    = {Mohammad Zeineldeen and
               Aleksandr Glushko and
               Wilfried Michel and
               Albert Zeyer and
               Ralf Schl{\"{u}}ter and
               Hermann Ney},
  editor    = {Hynek Hermansky and
               Honza Cernock{\'{y}} and
               Luk{\'{a}}s Burget and
               Lori Lamel and
               Odette Scharenborg and
               Petr Motl{\'{\i}}cek},
  title     = {Investigating Methods to Improve Language Model Integration for Attention-Based
               Encoder-Decoder {ASR} Models},
  booktitle = {Interspeech 2021, 22nd Annual Conference of the International Speech
               Communication Association, Brno, Czechia, 30 August - 3 September
               2021},
  pages     = {2856--2860},
  publisher = {{ISCA}},
  year      = {2021},
  url       = {https://doi.org/10.21437/Interspeech.2021-1255},
  doi       = {10.21437/Interspeech.2021-1255},
  timestamp = {Mon, 14 Mar 2022 16:42:12 +0100},
  biburl    = {https://dblp.org/rec/conf/interspeech/ZeineldeenGMZSN21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
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
