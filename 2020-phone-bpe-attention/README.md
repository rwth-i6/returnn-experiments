Paper title: A Systematic Comparison of Grapheme-based vs. Phoneme-based Label Units for Encoder-Decoder-Attention Models

Authors: Mohammad Zeineldeen, Albert Zeyer, Wei Zhou, Thomas Ng, Ralf Schlüter, Hermann Ney

Cite as:

```
@InProceedings { zeineldeen20:phon-att,
  author= {Zeineldeen, Mohammad and Zeyer, Albert and Zhou, Wei and Ng, Thomas and Schlüter, Ralf and Ney, Hermann},
  title= {A Systematic Comparison of Grapheme-based vs. Phoneme-based Label Units for Encoder-Decoder-Attention Models},
  booktitle= {IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year= 2021,
  month= may,
  note= {Submitted To},
  booktitlelink= {https://2021.ieeeicassp.org/},
  pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1153&row=pdf}
}
```

To reproduce our results, you would need to use [RETURNN](https://github.com/rwth-i6/returnn). 
Training the models, as well as decoding with the simplified decoder is done using RETURNN. 
For Switchboard, we used [RASR](https://github.com/rwth-i6/rasr) to extract gammatones features. 
For Librispeech, feature extraction is done using `librosa` in RETURNN. 
We also used RASR for the advanced decoder (prefix-tree label-synchronous decoder).
