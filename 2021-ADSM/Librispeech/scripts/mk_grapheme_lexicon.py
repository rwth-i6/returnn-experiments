#!/usr/bin/env python3
import sys, argparse
import lexicon_lib
import xml.etree.ElementTree as ET
import numpy as np
import itertools, math
import ProgressBar

# merge smaller grapheme units to larger one => new seqs and new grapheme label unit
def merge_seq(seq):
  assert merge > 0 or merge == -1
  seqList = []
  gs = seq.split()
  positions = range(len(gs)-1)
  if merge == -1: nMergeLimit = len(gs) # all possible merge until 1seg
  elif merge < 1: nMergeLimit = math.ceil(len(positions) * merge) + 1 # percentage
  else: nMergeLimit = min(int(merge)+1, len(gs))
  for nMerge in range(1, nMergeLimit):
    mergePos = list( itertools.combinations(positions, nMerge) )
    for mp in mergePos:
      s = ''
      for p in range(len(gs)):
        if p in mp: s += gs[p]
        else: s += gs[p] + ' '
      seqList.append(s.strip())
  return seqList

# filter invalid seqs
def filter_seq(seqs, min_nSeg, max_nSeg):
  # special symbol without acoustic realization: merge to neighbour segment
  # * remove in lexicon & corpus ?
  # * always separate as a single label ? (then max_nSeg += 1)
  special = ["-", "'", "_", "/"]

  seqList = []
  for seq in seqs:
    gs = seq.split()
    if len(gs) < min_nSeg or len(gs) > max_nSeg: continue
    skip = False
    for s in special:
      if s in gs: skip = True
    if not skip:
      seqList.append(seq)
  return seqList

# grapheme seqs from initial mapping + optional further merge
def get_mapped_graphemes(seqs, min_nSeg=1, max_nSeg=None):
  assert seqs
  if max_nSeg is None:
    max_nSeg = len(''.join(seqs[0].split()))
  gList = set(filter_seq(seqs, min_nSeg, max_nSeg))
  if not merge == 0:
    for seq in seqs:
      gList.update( filter_seq(merge_seq(seq), min_nSeg, max_nSeg) )
  return gList

# segment word into all possible grapheme segment
# - vocab: all allowed segmentations within the given label vocab + optional further merge
def get_all_graphemes(word, min_nSeg=1, max_nSeg=None, vocab=None):
  assert word, 'empty word %s ?' %word
  # as-is special word: e.g. [noise]
  if word.startswith('[') and word.endswith(']'):
    print('as-is special word %s' %word)
    return [word]

  # split word into allowed labels in vocabulary
  def recursive_split(seq):
    subwords = []
    for idx in range(1, len(seq)+1):
      if seq[:idx] in vocab:
        if idx == len(seq):
          subwords.append(seq)
        else:
          subList = recursive_split(seq[idx:])
          for s in subList:
            subwords.append(seq[:idx] + ' ' + s)
    return subwords
  
  # split word for specified positions
  def split_word(positions):
    subwords = []
    for pos in positions:
      sw = ''
      start = 0
      for p in pos:
        end = p + 1
        sw += word[start:end] + ' '
        start = end
      sw += word[start:]
      subwords.append(sw)
    return subwords

  gList = set()
  if max_nSeg is None:
    max_nSeg = len(word)
  if min_nSeg == 1 and not vocab:
    gList.add(word)
    min_nSeg += 1

  if vocab:
    gList.update( filter_seq(recursive_split(word), min_nSeg, max_nSeg) )
  else:
    positions = range(len(word) - 1)
    for nSeg in range(min_nSeg, max_nSeg+1):
      splitPos = list( itertools.combinations(positions, nSeg-1) )
      gList.update( filter_seq(split_word(splitPos), min_nSeg, max_nSeg) )
  # optional further merge
  if not merge == 0:
    for seq in list(gList):
      gList.update( filter_seq(merge_seq(seq), min_nSeg, max_nSeg) )
  return gList

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('in_lexicon')
  parser.add_argument('--out_lexicon', default='grapheme.lexicon')
  parser.add_argument('--corpus', default=None)
  parser.add_argument('--initial_vocab', default=None)
  parser.add_argument('--initial_mapping', default=None)
  # merge smaller units to larger one (advantage of text-based approach, e.g. word-piece)
  # -1: all possible merge until 1seg (word) | 0: no merge | < 1: percentage of length | >= 1: additional n merge (iterative approach)
  parser.add_argument('--merge', default=-1)
  # add EOW argumentation: same as w+ concatenation (different classes for within-word and word-end)
  parser.add_argument('--eow', action='store_true') # add to phon
  parser.add_argument('--eow_orth', action='store_true') # add to orth
  # oov words do not create new labels: only possible segmentation based on final vocab
  parser.add_argument('--restrict_oov', action='store_true')

  args = parser.parse_args(argv[1:])
  global merge
  merge = float(args.merge)

  # filter out words in the lexicon but not in the training corpus: labels can not be trained anyway
  # * possible non-segmentable words later ?
  # * always include single characters in the vocab: a theoretically full open-vocabulary (but still not trained)
  words = None
  if args.corpus is not None:
    words = set()
    import corpus_lib
    root2 = corpus_lib.get_corpus_root(args.corpus)
    # Note: find/findall can only be used for direct child nodes
    for orth in root2.getiterator('orth'):
      words.update(orth.text.split())
    print('uniq words in corpus:', len(words))
    for word in words:
      if len(word) > 50: # probably memory explosion for all possible split
        print('Warning: ierregular word', word)

  # initial vocab or mapping from AM/G2P alignment
  vocab = set()
  mapping = {}
  if args.initial_mapping is not None:
    # load initial mapping of word to possible grapheme sequences (can be merged to form new units)
    # Note: fixed format Word\t\tGraphemeSeq\t\tWeight\n
    nG = 0
    with open(args.initial_mapping, 'r') as f:
      for line in f.readlines():
        if not line.strip(): continue
        tokens = line.strip().split('\t\t')
        assert len(tokens) >= 2, 'invalid mapping %s' %line
        # Note: filter out by weight threshold should be done in mapping generation
        if not tokens[0] in mapping: mapping[tokens[0]] = []
        mapping[tokens[0]].append(tokens[1])
        nG += 1
        vocab.update(tokens[1].split())
    print('initial mapping: %d words and %d grapheme seqs' %(len(mapping), nG))
    print('initial label vocab size:', len(vocab))
  elif args.initial_vocab is not None:
    # load initial vocab and apply segmentation based on allowed units (can be merged to form new units)
    with open(args.initial_vocab, 'r') as f:
      data = f.read() # one label per line
      vocab = set(data.split())
    print('initial label vocab size:', len(vocab))

  # NOTE: include all single characters to be full open-vocabulary
  if vocab:
    import string
    for s in list(string.ascii_uppercase): # Librispeech: all upper cases
      vocab.add(s)
      if args.eow: vocab.add(s+'#')
      
  tree = lexicon_lib.parse_lexicon(args.in_lexicon)
  root = tree.getroot()
  labels = set()

  # loop over all lemmas
  nWords  = 0
  skipped = 0
  noMatch = 0
  nGraphemes = 0
  lemmaList  = root.findall('lemma')
  oovList = []

  def update_lemma(lemma, graphemes, restrict_vocab=False):
    for grapheme in sorted(graphemes):
      phon = ET.SubElement(lemma, 'phon')
      phon.text = grapheme
      if args.eow and not (grapheme.startswith('[') and grapheme.endswith(']')) and not grapheme.endswith('#'):
        phon.text += '#'
      if restrict_vocab:
        for p in phon.text.split():
          assert p in labels, '%s:%s in %s' %(p, phon.text, str(graphemes))
      else:
        labels.update(phon.text.split())
    # token element must appear after phon
    for tokName in ['synt', 'eval']:
      tok = lemma.find(tokName)
      if tok is not None:
        lemma.remove(tok)
        lemma.append(tok)

  # might take long: show status
  msg = ' Processing lemmas (total: %d) \n' %( len(lemmaList) )
  width = 40 if len(lemmaList)>=40 else len(lemmaList)
  progress_bar = ProgressBar.ProgressBar(msg, '=', width, len(lemmaList))
  progress_bar.restart()
  msg = ''

  for lemma in lemmaList:
    progress_bar.next()

    # special lemmas
    if lemma.attrib:
      # keep silence as blank
      if lemma.attrib['special'] == 'silence':
        phon = lemma.find('phon')
        assert phon is not None, 'silence has no phon tag'
        phon.text = '[blank]'
        # labels.add('[blank]') # move to later
      else:
        assert not lemma.findall('phon'), 'special lemma should not have pronunciation'
      continue

    # normal lemmas (words)
    orthList = lemma.findall('orth')
    assert orthList, 'no orthography for lemma ?'
    if words:
      skip = True
      for orth in orthList:
        if orth.text.strip() in words:
          skip = False
      if skip:
        root.remove(lemma)
        skipped += 1
        continue

    # first orth only (keep the rest alternatives for training usage) 
    word = orthList[0].text.strip()
    nWords += 1
    phons = lemma.findall('phon')
    minLen = np.inf
    maxLen = 0
    for phon in phons:
      # remove inserted blank between label repetitions 
      pron = phon.text.replace('[blank]','').replace('[SILENCE]','')
      nPhoneme = len(pron.split())
      if nPhoneme < minLen: minLen = nPhoneme
      if nPhoneme > maxLen: maxLen = nPhoneme
      lemma.remove(phon)

    # Assumption: each word is composed of n grapheme segments with acoustic structure (extreme case 1)
    # - max_nSeg => smaller grapheme unit
    #   - naive upper-bound: len(word)
    #   - max_nPhoneme: each segment has to cover at least 1 phoneme
    #     - non-acoustic grapheme (usually special symbol): merged into neighbouring segment
    # - min_nSeg => larger grapheme unit => group smaller units: independent of nPhonemes, should be data driven: both acoustic (model learning from speech) and text (transcription frequency)
    #   - naive lower-bound: 1 => really all possible segmentations but very expensive
    #   * possible simplification: some heuristics or G2P align ? (maybe not)
    if mapping and word in mapping:
      graphemes = get_mapped_graphemes(mapping[word], min_nSeg=1, max_nSeg=len(word))
    elif vocab:
      # only activate if subword labels also contain '#' already
      if args.eow_orth and not word.endswith('#'): word += '#'
      # oov w.r.t. mapping: restrict segmentation within new vocab without creating new labels
      if args.restrict_oov and mapping:
        oovList.append((word, minLen, maxLen, lemma))
        continue
      # all possible within-vocab segmentations
      graphemes = get_all_graphemes(word, min_nSeg=1, max_nSeg=maxLen, vocab=vocab)
      # spelling much longer than pronunciation
      # with all single chars, this should always succeed, unless special symbols in word
      if not graphemes and len(word) > maxLen:
        graphemes = get_all_graphemes(word, min_nSeg=1, max_nSeg=len(word), vocab=vocab)
    else:
      graphemes = get_all_graphemes(word, min_nSeg=minLen, max_nSeg=maxLen)

    if not graphemes:
      noMatch += 1
      msg += 'no matching grapheme sequence for word %s (max_nSeg: %d)\n' %(word, maxLen)
      # can not be segmented with all characters even: skip special symbol words (no phon)
      if vocab: continue

      if len(word) < minLen or merge == -1: minLen = 1
      elif merge > 0 and merge < 1: minLen = max(1, int(minLen * (1-merge)))
      elif merge >= 1: minLen = max(1, minLen - int(merge))
      graphemes = get_all_graphemes(word, min_nSeg=minLen, max_nSeg=maxLen)
      if not graphemes:
        graphemes = get_all_graphemes(word, min_nSeg=1, max_nSeg=maxLen)
        assert graphemes

    nGraphemes += len(graphemes)
    update_lemma(lemma, graphemes)

  progress_bar.finish()
  print(msg)

  if oovList:
    print('OOVs:', len(oovList), '  label-vocab size:', len(labels))
    merge = 0
    for word, minLen, maxLen, lemma in oovList:
      if args.eow and not word.endswith('#'): word += '#'
      graphemes = get_all_graphemes(word, min_nSeg=1, max_nSeg=maxLen, vocab=labels)
      if not graphemes:
        print('Warning: remove special symbol in word %s for segmentation' %word)
        word = word.replace("'",'') # hard-coded for BUSH' in LibriSpeech
        graphemes = get_all_graphemes(word, min_nSeg=1, max_nSeg=maxLen, vocab=labels)
      assert graphemes, word
      nGraphemes += len(graphemes)
      update_lemma(lemma, graphemes, restrict_vocab=True)

  # update label inventory
  phonemeInventory = root.find('phoneme-inventory')
  phonemeInventory.clear()
  for label in ['[blank]'] + sorted(labels):
    phoneme = ET.SubElement(phonemeInventory, 'phoneme')
    symbol = ET.SubElement(phoneme, 'symbol')
    symbol.text = label
    # only blank ? for grapheme probably all 
    variation = ET.SubElement(phoneme, 'variation')
    variation.text = 'none'
  labels.add('[blank]')

  msg = 'label vocab size (including [blank]):  %d' %len(labels)
  if vocab: msg += '  (%d from initial vocab)' %(len(labels.intersection(vocab)))
  print(msg)
  print('words: %d  (%d skipped from original lexicon)  (no match: %d)' %(nWords, skipped, noMatch))
  print('grapheme sequences: %d' %nGraphemes)

  lexicon_lib.pretty_print(tree, args.out_lexicon)
  print('Finished: %s\n' %args.out_lexicon)


if __name__ == '__main__':
  main(sys.argv)


