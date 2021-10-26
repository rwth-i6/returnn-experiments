#!/usr/bin/env python3

import sys

alignFile = sys.argv[1]
vocabFile = sys.argv[2]
mappingFile = sys.argv[3]

# grapheme label units
vocab = set()
# word to grapheme sequence
mapping = {}
with open(alignFile, 'r') as f:
  for line in f.readlines():
    if not line.strip(): continue
    tokens = line.split()
    gSeq = ''
    right_merge = False
    for idx in range(len(tokens)):
      gp = tokens[idx].split('}')
      assert len(gp) == 2, 'invalid align %s' %(tokens[idx])
      g = ''.join(gp[0].split('|'))
      p = ''.join(gp[1].split('|'))
      # skip epsilon
      if g == '_': continue
      # non-acoustic grapheme: merge with neighbor (left_merge priority except seq begin)
      if p == '_':
        if idx == 0: right_merge = True
        if right_merge:
          gSeq += g
        else: # left merge
          gSeq = gSeq.strip() + g + ' '
      else:
        if right_merge: right_merge = False
        gSeq += g + ' '
 
    graphemes = gSeq.split()
    word = ''.join(graphemes)
    vocab.update(graphemes)
    if word not in mapping:
      mapping[word] = set()
    mapping[word].add(' '.join(graphemes))

with open(vocabFile, 'w') as f:
  for v in sorted(vocab):
    f.write(v+'\n')

with open(mappingFile, 'w') as f:
  # Note: fixed format Word\t\tGraphemeSeq\n
  for word, gList in sorted(mapping.items()):
    for g in sorted(gList):
      f.write(word+'\t\t'+g+'\n')

