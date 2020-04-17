Configuration files for the experiments from Table 6 in the paper:

### Final results. Last lines show the results from the literature.

| Model | Features | Param. num. | Time [H:MM] | Spec Augm. | SWB WER [%] | CH WER [%] | Total WER [%] |
| :----------- | ----------: | ----------: | -----------: | -----------: | -----------: | ----------: | -----------: | 
| ResNet | LogMel ∆+∆∆ | 36.7M | 0:37 | no | 10.6 | 22.0 | 16.3 |
| ResNet | LogMel ∆+∆∆ | 36.7M | 0:37 | yes | 10.5 | 21.1 | 15.8 |
| LACE | LogMel ∆+∆∆ | 65.5M | 1:06 | no | 10.5 | 21.6 | 16.1 |
| LACE | LogMel ∆+∆∆ | 65.5M | 1:06 | yes | 9.8 | 19.8 | 14.8 |
| BLSTM | LogMel ∆+∆∆ | 41.1M | 1:03 | no | 10.0 | 19.2 | 15.0 |
| BLSTM | GT | 41.1M | 0:42 | no | 10.0 | 19.2 | 14.6 |
| BLSTM | GT | 41.1M | 0:42 | yes | 9.6 | 19.0 | 14.3 |
| [ResNet](https://arxiv.org/pdf/1703.02136.pdf) | LogMel ∆+∆∆ | - | - | no | 11.2 | — | — |
| [LACE](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/DeepCNNWithAttention-Interspeech2016.pdf) | LogMel  | - | - | no | 11.0 | — | — |
