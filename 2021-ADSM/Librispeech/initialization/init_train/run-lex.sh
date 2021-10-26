#!/usr/bin/bash
PY=python3

# cv is subset of dev sets
corpus=train-cv.corpus.xml

# lexicon: subword inventory + word-subwords dictionary
# vocab: all single characters already included, thus full open-vocabulary
# cv words always all possible within-vocabulary segmentation
in_lexicon=phon.train-cv.lexicon.xml
out_lexicon=vocab.train-cv.lexicon.xml
$py ../../scripts/mk_grapheme_lexicon.py $in_lexicon --corpus $corpus --out_lexicon $out_lexicon --initial_vocab ../g2p-align/3g-7p.vocab --merge 0 --eow | tee lex.log

# ensure CTC topology: force blank between label repetition
$py ../../scripts/lexicon-force-blank.py $out_lexicon train-cv.lexicon.xml

