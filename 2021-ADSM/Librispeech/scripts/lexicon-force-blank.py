#!/usr/bin/env python3
import sys
import lexicon_lib

# insert [blank] between label repetitions: automaton creation (training/alignment)
inLex = sys.argv[1]
outLex = sys.argv[2]
if len(sys.argv) >= 4:
  blank = sys.argv[3]
else: blank = '[blank]'

if len(sys.argv) >= 5:
  eow = bool(sys.argv[4])
else: eow = False

print('parse lexicon:', inLex)
tree = lexicon_lib.parse_lexicon(inLex)
root = tree.getroot()

for phon in root.getiterator('phon'):
  phonList = phon.text.split()
  if eow: phonList[-1] += '#'
  pron = ''
  for idx in range(len(phonList)-1):
    pron += phonList[idx] + ' '
    if phonList[idx] == phonList[idx+1]:
      pron += blank + ' '
  if eow: phonList[-1] = phonList[-1][:-1]
  pron += phonList[-1]
  phon.text = pron.strip()

lexicon_lib.pretty_print(tree, outLex)
print('Finished:', outLex) 

