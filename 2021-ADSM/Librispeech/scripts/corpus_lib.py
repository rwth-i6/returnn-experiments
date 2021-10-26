#!/usr/bin/env python3

import sys, os
import gzip
import xml.etree.ElementTree as ET

def parse_corpus( corpus ):
  assert os.path.exists(corpus), "corpus file %s not found !" %(corpus)
  if corpus.endswith('.gz'):
    tree = ET.parse( gzip.open(corpus) )
  else:
    tree = ET.parse(corpus)
  return tree

def get_corpus_root( corpus ):
  tree = parse_corpus(corpus)
  return tree.getroot()

def get_segments( corpus ):
  root = get_corpus_root(corpus)
  segL = []
  for seg in root.getiterator('segment'):
    segL.append(seg)
  return segL

def get_vocab(corpus):
  root = get_corpus_root(corpus)
  vocab = set()
  for orth in root.getiterator('orth'):
    vocab.update(orth.text.split())
  return vocab

def get_segment_list( corpus ):
  root = get_corpus_root(corpus)
  corpusName = root.attrib['name']
  segList = list()
  for rec in root.getiterator('recording'):
    recName = rec.attrib['name']
    recPath = rec.attrib['audio']
    segIdx=1
    for seg in rec.getiterator('segment'):
      if 'name' in seg.attrib.keys():
        segName = corpusName+'/'+recName+'/'+seg.attrib['name']
      else:
        segName = corpusName+'/'+recName+'/'+str(segIdx)
      segList.append( (segName, recPath) )
      segIdx += 1
  return segList

def get_segment_dict( corpus, lower=False, orthElement=False ):
  if isinstance(corpus, str):
    root = get_corpus_root(corpus)
  else: # can also be ElementTree
    assert isinstance(corpus, ET.ElementTree)
    root = corpus.getroot()
  corpusName = root.attrib['name']
  segDict = {}
  for rec in root.getiterator('recording'):
    recName = rec.attrib['name']
    segIdx=1
    for seg in rec.getiterator('segment'):
      if 'name' in seg.attrib.keys():
        segName = corpusName+'/'+recName+'/'+seg.attrib['name']
      else:
        segName = corpusName+'/'+recName+'/'+str(segIdx)
      assert segName not in segDict
      if orthElement:
        segDict[segName] = seg.find('orth')
      else:
        segDict[segName] = seg.find('orth').text.strip()
        if lower: segDict[segName] = segDict[segName].lower()
      segIdx += 1
  return segDict

def pretty_print(tree, corpusFile):
  if tree is not None:
    tree.write(corpusFile, encoding='UTF-8', xml_declaration=True)
  cmd = "xmllint --format %s -o %s" %(corpusFile, corpusFile)
  os.system(cmd)

