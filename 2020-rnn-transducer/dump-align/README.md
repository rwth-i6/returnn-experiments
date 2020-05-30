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
