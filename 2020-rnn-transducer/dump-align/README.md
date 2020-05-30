The `dump-align.py` script gets some RETURNN config (e.g. CTC model) and produces a Viterbi alignment,
over the full train + dev data (not sub epoch).

The script can optionally make use of a prior (in hybrid HMM fashion).
You can use the script `extract-softmax-prior.py` to get the average softmax (averaged over train),
which you can use as a prior.
(When you use a prior, the alignment would becomes less or not at all peaky.)

You need to create a couple symlinks here:

* `data`: Some dir where the resulting files will be saved to.
* `returnn`: Symlink to RETURNN.

See `notes.txt` for random custom notes on alignments which were created by us.
The alignments are not published here, but the training configs are all there, and it should be simple to reproduce.

Used alignments in the paper:

* CTC-align 4l: name `ctcalignfix-ctcalign-p0-4la`, config `ctcalign.prior0.lstm4la.withchar.lrkeyfix`
* CTC-align 6l: name `ctcalignfix-ctcalign-p0-6l`, config `ctcalign.prior0.lstm6l.withchar.lrkeyfix`
* CTC-align 6l with prior (non-peaky): name `ctcalignfix-ctcalign-p0-6l-extprior`, config `ctcalign.prior0.lstm6l.withchar.lrkeyfix` (+ prior)
* CTC-align 6l, less training: name `ctcalignfix-ctcalign-p0-6l-lrd05-prep2`, config `ctcalign.prior0.lstm6l.with-small-lstm.withchar.lrkeyfix.lrd05.pretrain-rep2.devtrain`
* Att. + CTC-align: name `ctcalignfix-base2-150`, config `base2.conv2l.specaug4a.ctc.devtrain`
* Transducer-align: name `rna-align-blank0-scratch-swap`, config `rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50`
