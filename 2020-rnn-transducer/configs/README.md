`*.config` files are [RETURNN config files](https://github.com/rwth-i6/returnn).

* Transducer model, RNA label topology, full-sum training: `rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`. Corresponds to Table 2, first line in the paper. In sub-ep 148 it gets 17.5% WER on Hub500 avg.
* Transducer model, RNA label topology, viterbi training: `rna-tf2.vit.ctcalignfix-ctcalign-norepfix-4l.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`
* Transducer model, CTC label topology, full-sum training: `rna-tf2.blank0.rep.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`
* Transducer model, CTC label topology, viterbi training: `rna-tf2.rep.vit.ctcalignfix-ctcalign-p0-4la.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`
* Transducer model, RNN-T label topology, full-sum training: `rnnt-warp.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config`

The transducer full-sum pure TF implementation (for RNA and CTC label topology) is in [`code/rna_tf_impl.py`](code/rna_tf_impl.py).

An exact mapping of all experiments in the paper (every line in all the tables) to all the config files
can be found in [_copy_configs.py](_copy_configs.py).

The final best config is [this](rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain.retrain1.config).
