`*.config` files are [RETURNN config files](https://github.com/rwth-i6/returnn).

The following table documents which files correspond to which model:

| Model in paper | Config                                                                                                                     |
| -------------  |                                                                                                                            |
| RNA full-sum   | `rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`                               |
| RNA viterbi    | `rna-tf2.vit.ctcalignfix-ctcalign-norepfix-4l.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config` |
| CTC full-sum   | `rna-tf2.blank0.rep.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`                           |
| CTC viterbi    | `rna-tf2.rep.vit.ctcalignfix-ctcalign-p0-4la.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`  |
| RNN-T full-sum | `rnnt-warp.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`                             |


