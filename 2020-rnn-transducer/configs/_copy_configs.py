#!/usr/bin/env python3

"""
Here we copy all relevant copies from our internal filesystem to this public repo.
This is for the paper [A New Training Pipeline for an Improved Neural Transducer](https://arxiv.org/abs/2005.09319).

In our internal filesystem, the relevant paths are here:

/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k
/u/zeyer/setups/switchboard/2020-03-12--sis
/u/merboldt/setups/2020-01-08--rnnt-rna

The Latex code of the paper can be found here:
/u/zeyer/Documents-archive/2020-rnnt-paper
(Or also in the [Arxiv -> download other formats -> source](https://arxiv.org/abs/2005.09319).)
In this Latex code, all the used configs files for each experiment are documented.
(Although we will also document all of that here publicly in the README.)

"""

import better_exchook
better_exchook.install()


base_dirs = [
    "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k",
    "/u/zeyer/setups/switchboard/2020-03-12--sis",
    "/u/merboldt/setups/2020-01-08--rnnt-rna",
]

configs = [
# 	\label{tab:swb:training-speed}, Table 1

#% RNA
"rna-tf4.3c-lm4a.convtrain.l2a_1e_4.encbottle256.attwb5_am.dec1la-n128.decdrop03.pretrain_less2_rep6.mlr50.emit2.encctc",
#% num params: 147260572
#% average epoch time 'GeForce GTX 1080 Ti': 0:47:34
#% in logdir: $ grep -e "[0-9]\+, finished after [0-9]* steps," train* | tail -71 | sort -k7r
#% fastest epoch: epoch 63, finished after 3742 steps, 0:51:00 elapsed (99.4% computing time)
#% 51:00 * 6
#		\multirow{5}{*}{Transd.} & RNA & \multirow{4}{*}{FS} & \multirow{3}{*}{TF} & \multirow{5}{*}{147} & 306 \\ \cline{2-2}\cline{6-6}
		
#% RNA-rep (CTC)
"rna-tf4.rep.3c-lm4a.convtrain.l2a_1e_4.encbottle256.attwb5_am.dec1la-n128.decdrop03.pretrain_less2_rep6.mlr50.emit2.encctc",
#% num params: 147260572
#% average epoch time 'GeForce GTX 1080 Ti': 0:54:50
#% fastest epoch: epoch 142, finished after 3743 steps, 0:54:24 elapsed (99.3% computing time)
#		& CTC &  &  &  & 326 \\ \cline{2-2}\cline{6-6}

#% RNN-T (TF)
"rnnt-tf3.3c-lm4a.convtrain.l2a_1e_4.encbottle256.attwb5_am.dec1la-n128.decdrop03.pretrain_less2_rep6.mlr50.emit2.encctc",
#% num params: 147260572
#% grep "average epoch" scores/rnnt-tf3.3c-lm4a.convtrain.l2a_1e_4.encbottle256.attwb5_am.dec1la-n128.decdrop03.pretrain_less2_rep6.mlr50.emit2.encctc.train.info.txt 
#% average epoch time 'GeForce GTX 1080 Ti': 0:48:26
#% train.o1112897.1:pretrain epoch 38, finished after 3732 steps, 0:55:30 elapsed (99.3% computing time)
#	& \multirow{2}{*}{RNN-T} &  &  &  & 333\\ \cline{4-4}\cline{6-6}

#% RNN-T (warp-transducer)
"rnnt-warp.3c-lm4a.convtrain.l2a_1e_4.encbottle256.attwb5_am.dec1la-n128.decdrop03.pretrain_less2_rep6.mlr50.emit2.encctc",
#% (*6) = 218.6
#% num params: 147260572
#% average epoch time 'GeForce GTX 1080 Ti': 0:33:33 
#% fastest epoch: epoch 73, finished after 3724 steps, 0:36:26 elapsed (98.9% computing time)
#	&  &  & CUDA &  & 219\\ \cline{2-4}\cline{6-6}

#% RNA (viterbi)
"rna3c-lm4a-noatt.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-lintanh-n128-noprevout.decdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain",
#% num params: 147259541
#% average epoch time 'GeForce GTX 1080 Ti': 0:24:10 (159.7 minutes/epoch)
#% fastest epoch: epoch  96,   finished  after  3614  steps,  0:26:37  elapsed  (99.2%  computing  time)
#& CTC & \multirow{2}{*}{CE} & \multirow{2}{*}{TF} &  & 160\\

#% Attention baseline
"base2.conv2l.specaug4a.ctc.devtrain",
#% num params: 161641308
#% average epoch time 'GeForce GTX 1080 Ti': 0:18:29 (*6=138.3)
#% fastest epoch: epoch  134,  finished  after  1804  steps,  0:23:05  elapsed  (99.5%  computing  time)
#Attention & $-$ &  &  &  162  & 138 \\


# ---------------
# 	\label{tab:swb:fullsum-vs-approx}, Table 2
"rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50",
#% Epoch    Hub5'00 Swb    Hub5'00 CH    Hub5'00 Avg    Hub5'01
#% 148           11.5          23.4           17.5       16.5
#		\multirow{2}{*}{RNA} & FS  & 11.5 & 23.4 & 17.5 & 16.5 \\ \cline{2-6}

"rna-tf2.vit.ctcalignfix-ctcalign-norepfix-4l.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50",
#% alignment ctcalignfix-ctcalign-norepfix-4l
#% 150 (hub5_00_avg=15.2  hub5_00_swb=10.1  hub5_00_ch=20.4  hub5_01=14.8 ) 
#& Vit. & 10.1 & 20.4 & 15.2 & 14.8\\

#% andre:
"rna-tf2.blank0.rep.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50",
#% (hub5_00_avg=19.8  hub5_00_swb=15.0  hub5_00_ch=24.6  hub5_01=20.1 )
#		\multirow{2}{*}{CTC} & FS  & 15.0 & 24.6 & 19.8 & 20.1\\ \cline{2-6}

"rna-tf2.rep.vit.ctcalignfix-ctcalign-p0-4la.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50", #.recog.wers.txt
#% (hub5_00_avg=15.6  hub5_00_swb=10.5  hub5_00_ch=20.6  hub5_01=15.3 )
#		& Vit. & 10.5 & 20.6 & 15.6 & 15.3\\

"rnnt-warp.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50",
#% (hub5_00_avg=17.0  hub5_00_swb=11.6  hub5_00_ch=22.3  hub5_01=16.4 )
#		RNN-T & FS   & 11.6 & 22.3 & 17.0 & 16.4\\


# ---------------
#   \label{tab:swb:alignments}, Table 3

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-4la.chunk60.encctc.devtrain", # 14.7
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-4la.chunk60.encctc.devtrain", # 14.3
# CTC-align 4l & 14.7 & 14.3 \\ \hline

#  baseline:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7 (not 14.5)
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
# CTC-align 6l & 14.7 & 14.5 \\ \hline

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l-extprior.chunk60.encctc.devtrain", # (non-peaky) 15.4
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l-extprior.chunk60.encctc.devtrain", # (non-peaky) 14.9
# CTC-align 6l with prior (non-peaky) & 15.4 & 14.9 \\ \hline

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l-lrd05-prep2.chunk60.encctc.devtrain", # (less training, only 60 ep) 14.6
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l-lrd05-prep2.chunk60.encctc.devtrain", # (less training, only 60 ep) 14.6
# CTC-align 6l, less training & 14.6 & 14.6 \\ \hline

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-base2-150.chunk60.encctc.devtrain", # 14.4
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-base2-150.chunk60.encctc.devtrain", # 14.2
# Att. + CTC-align & 14.4 & 14.2 \\ \hline

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain", # 14.2
# % ep142: (hub5_00_avg=14.2  hub5_00_swb=9.4   hub5_00_ch=19.0  hub5_01=13.8 )
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.rna-align-blank0-scratch-swap.chunk60.encctc.devtrain", # 14.1
# Transducer-align & 14.2 & 14.1 \\


# ----------
# \label{sec:exp:variations}
# (also in next table)

# baseline B1:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7
# baseline B2:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5


# ---------
#   \label{tab:swb:ablation}, Table 4

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
#    Baseline & 14.7 & 14.5 \\ \hline

#     no chunk: 
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.encctc.devtrain", # 16.3
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.encctc.devtrain", # 15.7
# No chunked training & 16.3 & 15.7 \\ \hline

#     no switchout:
"rna3c-lm4a.convtrain.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 15.0
"rna3c-lm4a.convtrain.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
#    No switchout & 15.0 & 14.5 \\ \hline

#  Fast LM / no LM mask:
"rna3c-lm4a-nomask.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.8
"rna3c-lm4a-nomask.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.8
# SlowRNN always updated (not slow) & 14.8 & 14.8 \\ \hline

"rna3c-nolm.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n1024.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.8
"rna3c-nolm.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n1024.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7
#  No SlowRNN & 14.8 & 14.7 \\ \hline

#        no att:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
#    No attention & 14.5 & * \\ \hline

# bigger dec LSTM:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n512.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.3
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n512.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7
#   FastRNN dim 128 $\rightarrow$ 512 & 14.3 & 14.7 \\ \hline

#  really no prev:out:
"rna3c-lm4a-noatt.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128-noprevout.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.9
"rna3c-lm4a-noatt.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128-noprevout.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
# No label feedback & 14.9 & 14.5 \\ \hline

# just noatt in lm:
"rna3c-lm4a-noatt.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.9
"rna3c-lm4a-noatt.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
# No encoder feedback to SlowRNN & 14.9 & 14.5 \\ \hline

# no sep emit:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.9
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.9
#    No separate blank sigmoid & 14.9 & 14.9 \\


# -----------------------
#   \label{tab:swb:enc-import}, Table 5

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.7 (not 14.5)
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.5
# None & 14.7 & 14.5 \\ \hline

"rna3c-ctcalign-p0-6l-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.mlr50.emit2.fl2.rep.fixmask.ctcalignfixsame.chunk60.encctc.devtrain", # 15.4
# ctcalign-6l import:
"rna3c-ctcalign-p0-6l-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.mlr50.emit2.fl2.rep.fixmask.ctcalignfixsame.chunk60.encctc.devtrain", # 15.5
# CTC as encoder & 15.4 & 15.5 \\ \hline

"rna3c-base2_150-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 14.2
#     base2-150 import:
"rna3c-base2_150-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain", # 13.9
# Att. encoder & 14.2 & 13.9 \\ \hline

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1", # 13.7
# retrain1:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1", # 13.6
# Transducer (itself) & 13.7 & 13.6 \\


# -----------------------
# 	\label{tab:swb:beam-sizes}, Table 6

# (These are mostly variations of recog settings...)
"base2.conv2l.specaug4a.ctc.devtrain.retrain1",
# RNA:
# "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l....gMy3zUnBdiyy",
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1",


# -----------------------
# 	\label{tab:swb:longer-seqs}, Table 7

# att: output/extern.base2.conv2l.specaug4a.ctc.devtrain.retrain1.concat30.txt
"base2.conv2l.specaug4a.ctc.devtrain.retrain1",
# RNA: extern.rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l....gMy3zUnBdiyy
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1",
# TODO we should publish the concat seq script...


# ------------------------
#   \label{tab:swb:overall}, Table 8

#            base2.conv2l.specaug4a.ctc.devtrain 15.2, ep 146
"base2.conv2l.specaug4a.ctc.devtrain",
# output/experiment.extern.base2.conv2l.specaug4a.ctc.devtrain.summary.txt
#Checked epochs: [146, 148, 149, 150]
#Select best epoch by dataset: ['hub5e_00', 'hub5e_00: Callhome']
#Best epoch: 146
#hub5e_00: Callhome: 21.1
#hub5e_00: Switchboard: 9.2
#hub5e_00: 15.2
#hub5e_01: Switchboard: 10.4
#hub5e_01: Switchboard-2 Phase III: 13.5
#hub5e_01: Switchboard-Cell: 18.5
#hub5e_01: 14.2
#rt03s: Swbd: 21.2
#rt03s: Fisher: 13.7
#rt03s: 17.6
#    \multirow{2}{*}{Att.} & \multirow{4}{*}{BPE} & \multirow{4}{*}{1k} &  \phantom{0}25 & \multirow{4}{*}{no} & 9.2 & 21.1 & 15.2 & 14.2 & 17.6  \\

#            base2.conv2l.specaug4a.ctc.devtrain.retrain1 14.1
"base2.conv2l.specaug4a.ctc.devtrain.retrain1",
# output/experiment.extern.base2.conv2l.specaug4a.ctc.devtrain.retrain1.summary.txt
#Checked epochs: [40, 80, 141, 142, 144, 148, 149, 150]
#Select best epoch by dataset: ['hub5e_00', 'hub5e_00: Callhome']
#Best epoch: 150
#hub5e_00: Callhome: 19.3
#hub5e_00: Switchboard: 8.7
#hub5e_00: 14.0
#hub5e_01: Switchboard: 9.3
#hub5e_01: Switchboard-2 Phase III: 12.5
#hub5e_01: Switchboard-Cell: 17.8
#hub5e_01: 13.3
#rt03s: Swbd: 20.1
#rt03s: Fisher: 12.9
#rt03s: 16.6
#            base2.conv2l.specaug4a.ctc.devtrain.retrain1.keeplast20 14.0, ep 143
#"base2.conv2l.specaug4a.ctc.devtrain.retrain1.keeplast20",
#    & &&  \phantom{0}50 & & 8.7 & 19.3 & 14.0 & 13.3 & 16.6 \\

# merboldt:
"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain", # 149
# 149 (hub5_00_avg=14.1  hub5_00_swb=9.4   hub5_00_ch=18.7  hub5_01=14.1 )
#rt03s: Swbd: 20.2
#rt03s: Fisher: 12.9
#rt03s: 16.7
#\multirow{2}{*}{Transd.}    & && \phantom{0}25 & & 9.4 & 18.7 & 14.1 & 14.1 & 16.7\\

"rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain.retrain1", # .145
# Epoch    Hub5'00 Swb    Hub5'00 CH    Hub5'00 Avg    Hub5'01     RT03S
# 145            8.7          18.3           13.5       13.3       15.6
#    & && \phantom{0}50 & & 8.7 & 18.3 & 13.5 & 13.3 & 15.6 \\

]


