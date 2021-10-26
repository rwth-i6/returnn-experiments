#!/usr/bin/env bash
PY=python3

# not really used: just verify all words in the out lexicon
corpus=../../initialization/init_train/train-cv.corpus.xml

# Viterbi aligning the training data with the trained model (prior) + weight filtering (e.g. 0.05)
# Note: all the single characters should already be included (in most cases)
train_map=train.map # for example

# lexicon: subword inventory + word-subwords dictionary
# subword merging
# - train words: map + merge => vocab
# - cv oovs: all possible segmentation based on vocab
in_lexicon=../../initialization/init_train/train-cv.lexicon.xml
out_lexicon=map.train-cv.lexicon.xml
$py ../../scripts/mk_grapheme_lexicon.py $in_lexicon --corpus $corpus --out_lexicon $out_lexicon --initial_mapping $train_map --merge 1 --eow --restrict_oov | tee lex.log

# ensure CTC topology: force blank between label repetition
$py ../../scripts/lexicon-force-blank.py $out_lexicon train-cv.lexicon.xml

# Repeat Iteration: again vocab refinement with subsampling+2
# Note: also adjust num_outputs (classes) in crnn.config (RETURNN) and lexicon in sprint config (RASR)

