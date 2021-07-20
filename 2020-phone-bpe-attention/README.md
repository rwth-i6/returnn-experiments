Paper title: A Systematic Comparison of Grapheme-based vs. Phoneme-based Label Units for Encoder-Decoder-Attention Models

Authors: Mohammad Zeineldeen, Albert Zeyer, Wei Zhou, Thomas Ng, Ralf Schlüter, Hermann Ney

Cite as:

```
@Misc {zeineldeen20:phon-att,
  title= {A Systematic Comparison of Grapheme-based vs. Phoneme-based Label Units for Encoder-Decoder-Attention Models},
  author= {Zeineldeen, Mohammad and Zeyer, Albert and Zhou, Wei and Ng, Thomas and Schlüter, Ralf and Ney, Hermann},
  month= nov,
  year= 2020,
  note= {Preprint arXiv:2005.09336 },
  pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1177&row=pdf},
  url = {http://arxiv.org/abs/2005.09336}
}
```

To reproduce our results, you would need to use [RETURNN](https://github.com/rwth-i6/returnn). 
Training the models, as well as decoding with the simplified decoder is done using RETURNN. 
For Switchboard, we used [RASR](https://github.com/rwth-i6/rasr) to extract gammatones features. 
For Librispeech, feature extraction is done using `librosa` in RETURNN. 
We also used RASR for the advanced decoder (prefix-tree label-synchronous decoder).
