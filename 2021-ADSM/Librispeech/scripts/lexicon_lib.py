#!/usr/bin/env python3

import os, sys
from collections import OrderedDict
import xml.etree.ElementTree as ET

def parseDictVocab( bpeFile ):
  f = open(bpeFile,'rU')
  lines = f.readlines()
  f.close()

  vocabDict = OrderedDict()
  for line in lines:  
    line = line.strip()
    if line.startswith('{') or line.startswith('}') or not line: continue
    v = line.split(':')[0].strip()[1:-1]
    i = line.split(':')[1].replace(",","").strip()
    if v in vocabDict.keys():
      print("Warning: duplicate entry %s index %s" %(v,i))
    vocabDict[v] = i
  return vocabDict

def makeDictVocab( vocabDict, vocabFile ):
  with open(vocabFile, 'w') as f:
    f.write('{\n')
    for v, i in sorted(vocabDict.items(), key=lambda item: int(item[1])):
      f.write('"%s": %s,\n' %(v, i))
    f.write('}')

def makeVocabFile( vocabDict, vocabFile ):
  with open(vocabFile, 'w') as f:
    for v in sorted(vocabDict.keys()):
      f.write(v+' '+vocabDict[v]+'\n')

def parseVocabFile( vocabFile ):
  vocabDict = {}
  with open(vocabFile, 'r') as f: 
    for line in f:
      if not line.strip(): continue
      tokens = line.split()
      assert len(tokens) >= 2, line
      vocabDict[tokens[0]] = ' '.join(tokens[1:])
  return vocabDict

def makeBaseLexiconXml(phonemeList=None, unkTok='<unk>', unkPron=None):
  root = ET.Element('lexicon')
  tree = ET.ElementTree(root)
  # list of phonemes(or bpes)
  if phonemeList is not None:
    pI = ET.SubElement(root, 'phoneme-inventory')
    for phon in phonemeList:
      phoneme = ET.SubElement(pI, 'phoneme')
      symbol = ET.SubElement(phoneme, 'symbol')
      symbol.text = phon
      #var = ET.SubElement(phoneme, 'variation')
      #var.text = 'none'
  else:
    comment = ET.Comment('no phoneme-inventory')
    root.append(comment)

  # special lemma #
  specialLemmaDict = [ ('sentence-begin', '<s>', None), 
                       ('sentence-end', '</s>', None), 
                       ('unknown', unkTok, unkPron) 
                     ] # no silence
  for spLemma, token, pron in specialLemmaDict:
    lemma = ET.SubElement(root, 'lemma', {"special": spLemma})
    orth = ET.SubElement(lemma, 'orth')
    #orth.text = '['+spLemma.upper()+']'
    orth.text = token
    if pron is not None:
      phon = ET.SubElement(lemma, 'phon')
      phon.text = pron
    synt = ET.SubElement(lemma, 'synt')
    if token is None:
      ET.SubElement(lemma, 'eval')
    else:
      tok = ET.SubElement(synt, 'tok')
      tok.text = token    

  return root, tree


def addVocabToLexicon(root, vocabDict):
  special = ['<s>','</s>','<unk>']
  for v in sorted(vocabDict.keys()):
    if v in special: continue
    lemma = ET.SubElement(root, 'lemma')
    orth = ET.SubElement(lemma, 'orth')
    orth.text = v
    # token is the same for now

def addVocabPronToLexicon(root, vocab2pron, output=None):
  if output is not None:
    roottext = root.replace('</lexicon>','')
    root = open(output, 'w')
    root.write(roottext + '\n')
    roottext = '</lexicon>'

  special = ['<s>','</s>','<unk>']
  for v in vocab2pron.keys():
    if v in special: continue
    # output to file directly (memory issue for large vocabulary)
    if output is not None:
      lemma = ET.Element('lemma')
    else:
      lemma = ET.SubElement(root, 'lemma')
    orth = ET.SubElement(lemma, 'orth')
    orth.text = v
    for phonemes in vocab2pron[v]:
      phon = ET.SubElement(lemma, 'phon')
      phon.text = ' '.join(phonemes)
    # token is the same for now

    if output is not None:
      lemmatext = ET.tostring(lemma, encoding='utf-8').decode()
      root.write(lemmatext + '\n')
  if output is not None:
    root.write(roottext)
    root.close()

def pretty_print(tree, lexFile):
  if tree is not None:
    tree.write(lexFile, encoding='UTF-8', xml_declaration=True)
  cmd = "xmllint --format %s -o %s" %(lexFile, lexFile)
  os.system(cmd)


def parse_lexicon(lexFile):
  if lexFile.endswith('.gz'):
    import gzip
    tree = ET.parse(gzip.open(lexFile, 'r'))
  else:
    tree = ET.parse(lexFile)
  return tree


# copy parts of Albert's code: find bpe sequences for word
class PrefixTree:
  """
  Prefix tree / trie.
  This class represents both a single node and the tree.
  """

  def __init__(self, prefix="", root=None, BpeMergeSymbol='@@'):
    """
    :param str prefix:
    :param PrefixTree|None root:
    """
    self.prefix = prefix
    self.arcs = {}  # type: typing.Dict[str,PrefixTree]
    self.finished = False
    self.bpe_finished = False
    self.is_root = not root
    self.root = root
    self.BpeMergeSymbol = BpeMergeSymbol

  def add(self, postfix, root=None):
    """
    :param str postfix:
    :param None|PrefixTree root:
    :rtype: PrefixTree
    """
    if not root:
      if self.is_root:
        root = self
      else:
        assert self.root
        root = self.root
    if postfix == self.BpeMergeSymbol:
      arc = postfix
      postfix_ = ""
    else:
      arc = postfix[:1]
      postfix_ = postfix[1:]
    if arc in self.arcs:
      child = self.arcs[arc]
    else:
      child = PrefixTree(root=root, prefix=self.prefix + arc)
      self.arcs[arc] = child
    if arc == self.BpeMergeSymbol and not postfix_:
      self.bpe_finished = True
    if postfix_:
      return child.add(postfix_, root=root)
    else:
      child.finished = True
      return child

class Hyp:
  """
  Represents a hypothesis in the search.
  """

  def __init__(self, bpe_sym_history, cur_node):
    """
    :param list[str] bpe_sym_history:
    :param PrefixTree cur_node:
    """
    self.bpe_sym_history = bpe_sym_history
    self.cur_node = cur_node

  def copy(self):
    pass


class Search:
  """
  Covers the search hyps and the search itself.
  """

  def __init__(self, bpe, word, word_pos=0):
    """
    :param PrefixTree bpe:
    :param str word:
    :param int word_pos:
    """
    self.bpe = bpe
    self.word = word
    self.word_pos = word_pos
    self.hyps = [Hyp(bpe_sym_history=[], cur_node=bpe)]  # type: typing.List[Hyp]
    self.final_bpe_seqs = None  # type: typing.Optional[typing.List[typing.List[str]]]

  def _get_finished(self):
    assert self.word_pos == len(self.word)
    finals = []  # type: typing.List[typing.List[str]]
    for hyp in self.hyps:
      if hyp.cur_node.finished:
        finals.append(hyp.bpe_sym_history + [hyp.cur_node.prefix])
    self.final_bpe_seqs = finals

  def _expand(self):
    assert self.word_pos < len(self.word)
    char = self.word[self.word_pos]
    new_hyps = []  # type: typing.List[Hyp]
    for hyp in self.hyps:
      if hyp.cur_node.bpe_finished:
        next_node = self.bpe.arcs.get(char)
        if next_node:
          new_hyps.append(
            Hyp(
              bpe_sym_history=hyp.bpe_sym_history + [hyp.cur_node.prefix + "@@"],
              cur_node=next_node))
      next_node = hyp.cur_node.arcs.get(char)
      if next_node:
        new_hyps.append(Hyp(bpe_sym_history=hyp.bpe_sym_history, cur_node=next_node))
    self.hyps = new_hyps

  def search(self):
    """
    :return: collection of possible BPE symbol seqs
    :rtype: list[list[str]]
    """
    while self.word_pos < len(self.word):
      self._expand()
      self.word_pos += 1
    self._get_finished()
    return self.final_bpe_seqs

