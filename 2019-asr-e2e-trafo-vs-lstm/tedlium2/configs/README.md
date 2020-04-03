
From the Tedlium table 4 in the paper:

* Transformer (14.6%): `trafo.specaug4.12l.ffdim2.pretrain3a.lr0008.ctc.dr0.2.wu10.config`
* Transformer + 36Enc (13.9%): `trafo.specaug4.enc36l.dec12l.n512.ffdim2.pretrain.lr0008.ctc.dr0.3.wu.acc3.config`
* LSTM (12.4%): `base2.specaug.bs18k.curric3.pfixedlr.lrd07.eos.config`
* LSTM + Conv (12.6%): `base2.conv2l.specaug.curric3.eos.config`
* LSTM + ExpV (11.7%): `base2.smlp2.specaug4.bs18k.curric3.eos.config`
* LSTM + ExpV + LM (10.5%): `base2.smlp2.specaug4.bs18k.curric3.eos.all.nopos_transfo_30_d0.4096_768.8h.lr1.cl1.config` (just for recognition, not for training)
* LSTM + ExpV + LM-EOS (10.3%): `base2.smlp2.specaug4.bs18k.curric3.eos.all.nopos_transfo_30_d0.4096_768.8h.lr1.cl1.tune_eos.config` (just for recognition, not for training)
* DecLSTM (18.3%): `declstmindep.specaug4.decl2.nowfb.dotatt.eos.config`
