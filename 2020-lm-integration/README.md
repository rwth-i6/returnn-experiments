Paper: [Early Stage LM Integration Using Local and Global Log-Linear Combination](https://arxiv.org/abs/2005.10049).

Authors: Wilfried Michel, Ralf Schlüter, Hermann Ney

The paper will be published at Interspeech 2020.

Cite as:
```
@InProceedings { michel20:lm_integration,
author= {Michel, Wilfried and Schl\"uter, Ralf and Ney, Hermann},
title= {Early Stage LM Integration Using Local and Global Log-Linear Combination},
booktitle= {Interspeech},
year= 2020,
address= {http://www.interspeech2020.org/},
month= oct,
note= {To appear},
pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1140&row=pdf}
}
```
(Citation will be updated when the paper gets published at Interspeech.)

---

These configs are based on the `base2.conv2l.specaug.curric3.config` which can be found [here](https://github.com/rwth-i6/returnn-experiments/blob/master/2019-asr-e2e-trafo-vs-lstm/librispeech/base2.conv2l.specaug.curric3.config) and were extended to use external LMs.

Variables starting with `ext_` are designed to be set by the external hyper parameter tuning engine. Some useful tuning hints can be found in the paper.
