Here are the configs for the paper
[CTC in the Context of Generalized Full-Sum HMM Training](https://www-i6.informatik.rwth-aachen.de/publications/download/1035/Zeyer--2017.pdf).

Please cite as:
```
@InProceedings { zeyer2017:ctc,
author= {Zeyer, Albert and Beck, Eugen and Schlüter, Ralf and Ney, Hermann},
title= {CTC in the Context of Generalized Full-Sum HMM Training},
booktitle= {Interspeech},
year= 2017,
pages= {944-948},
address= {Stockholm, Sweden},
month= aug,
booktitlelink= {http://www.interspeech2017.org/},
pdf = {https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1035&row=pdf}
}
```

The configs are
to be used with [RETURNN](https://github.com/rwth-i6/returnn)
(called CRNN earlier/internally)
and [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/)
(called Sprint internally)
for data preprocessing and decoding.

The experiments are done on the Switchboard 300h English corpus but we also cannot publish the data ourselves.

To use the RETURNN configs with other data,
replace the `train`/`dev` config settings, which specify the train and dev corpus data.
At the moment, they will use the `ExternSprintDataset` interface to get the preprocessed data out of RASR.
You can also use other dataset implementations provided by RETURNN (see RETURNN doc / source code),
e.g. the HDF format directly.

---

Some extensions on the experiments were performed later,
as part of the studies on some formal analysis on the peaky behavior.
The formal analysis was published as a separate paper
with configs [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2021-formal-peaky-behavior-ctc).
The extended Switchboard experiments can be found
in [`switchboard-extended2020`](switchboard-extended2020).
