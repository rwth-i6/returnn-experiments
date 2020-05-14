#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from xml.dom import minidom
import codecs
from returnn.LmDataset import Lexicon
from argparse import ArgumentParser

"""
create Lexicon, given bpe Vocab, lexicon and applied phones_bpe 
"""

def convert(string_num):
  if isinstance(string_num, str) and string_num.startswith("0"):
    return "zero " + convert(string_num[1:])

  num = int(string_num)  
  units = ("", "one ", "two ", "three ", "four ","five ", "six ", "seven ","eight ", "nine ", "ten ", "eleven ", "twelve ", "thirteen ", "fourteen ", "fifteen ","sixteen ", "seventeen ", "eighteen ", "nineteen ")
  tens =("", "", "twenty ", "thirty ", "forty ", "fifty ","sixty ","seventy ","eighty ","ninety ")
  if num<0:
    return "minus "+convert(-num)
  if num<20:
    return  units[num]
  if num<100:
    return  tens[num // 10]  +units[int(num % 10)] 
  if num<1000:
    return units[num // 100]  +"hundred " +convert(int(num % 100))
  if num<1000000: 
    return  convert(num // 1000) + "thousand " + convert(int(num % 1000))
  if num < 1000000000:    
    return convert(num // 1000000) + "million " + convert(int(num % 1000000))            
  return convert(num // 1000000000)+ "billion "+ convert(int(num % 1000000000))
                                                                                    
def hasNumber(inputString):
  return any(char.isdigit() for char in inputString)

def separate(iString):
  prev_char = iString[0]
  tmp = []
  new = iString[0]
  for x, i in enumerate(iString[1:]):
    if i.isalpha() and prev_char.isalpha():
      new += i
    elif i.isnumeric() and prev_char.isnumeric():
      new += i
    else:
      tmp.append(new)
      new = i
    prev_char = i
    if x == len(iString)-2:
      tmp.append(new)
      new = ''
  if len(iString) > 1:
      return tmp
  return [iString]

def to_unicode_list(input_l):
  res = []
  for item in input_l:
    res.append(to_unicode(item))
  return res

def to_unicode(input):
  text = input.split()
  result = ""
  for k in text:
    result += phone_to_unicode[k]
  return result

# map phone into unicode
phone_to_unicode = {'[LAUGHTER]': 'L',
                    '[NOISE]': 'N',
                    '[SILENCE]': 'S',
                    '[VOCALIZEDNOISE]': 'V',
                    'aa': 'a',
                    'ae': 'à',
                    'ah': 'á',
                    'ao': 'â',
                    'aw': 'ã',
                    'ax': 'ä',
                    'ay': 'å',
                    'b': 'b',
                    'ch': 'c',
                    'd': 'd',
                    'dh': 'ď',
                    'eh': 'e',
                    'el': 'è',
                    'en': 'é',
                    'er': 'ê',
                    'ey': 'ë',
                    'f': 'f',
                    'g': 'g',
                    'hh': 'h',
                    'ih': 'i',
                    'iy': 'ì',
                    'jh': 'j',
                    'k': 'k',
                    'l': 'l',
                    'm': 'm',
                    'n': 'n',
                    'ng': 'ñ',
                    'ow': 'o',
                    'oy': 'ò',
                    'p': 'p',
                    'r': 'r',
                    's': 's',
                    'sh': 'ś',
                    't': 't',
                    'th': 'ţ',
                    'uh': 'u',
                    'uw': 'ù',
                    'v': 'v',
                    'w': 'w',
                    'y': 'y',
                    'z': 'z',
                    'zh': 'ź',
                    ' ': ' ',
                    '#1': '#1', # disambiquate symbols for homophones
                    '#2': '#2',
                    '#3': '#3',
                    '#4': '#4',
                    '#5': '#5',
                    '#6': '#6',
                    '#7': '#7',
                    '#8': '#8',
                    '#9': '#9',
                    '#10': '#10',
                    '#11': '#11',
                    '#12': '#12',
                    '#13': '#13',
                    '#14': '#14',
                    }

def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--bpe_vocab", required=True)
  arg_parser.add_argument("--lexicon", required=True)
  arg_parser.add_argument("--phones_bpe", required=True)
  arg_parser.add_argument("--bpe", action="store_true")
  arg_parser.add_argument("--char", action="store_true")
  arg_parser.add_argument("--charbpe", action="store_true")
  arg_parser.add_argument("--disamb", action="store_true")
  arg_parser.add_argument("--output", required=True)
  args = arg_parser.parse_args()
  
  #if single char or phon need to comment the optional arg phones_bpe since if we dont use bpe
  bpe1k_file = args.bpe_vocab
  lexicon_file = args.lexicon
  phones_bpe_file = args.phones_bpe

  def create_specialTree(input):
    if input == "</s>":
      lemma = ET.SubElement(lex_root, 'lemma', special="sentence-end")
      orth = ET.SubElement(lemma, 'orth')
      synt = ET.SubElement(lemma, 'synt')
      tok = ET.SubElement(synt, 'tok')
      orth.text = '[SENTENCE-END]'
      tok.text = input
      eval = ET.SubElement(lemma, 'eval')
    elif input == "<s>":
      lemma = ET.SubElement(lex_root, 'lemma', special="sentence-begin")
      orth = ET.SubElement(lemma, 'orth')
      synt = ET.SubElement(lemma, 'synt')
      tok = ET.SubElement(synt, 'tok')
      orth.text = '[SENTENCE-BEGIN]'
      tok.text = input
      eval = ET.SubElement(lemma, 'eval')
    elif input == "<unk>":
      lemma = ET.SubElement(lex_root, 'lemma', special="unknown")
      orth = ET.SubElement(lemma, 'orth')
      synt = ET.SubElement(lemma, 'synt')
      tok = ET.SubElement(synt, 'tok')
      orth.text = '[UNKNOWN]'
      tok.text = input
      eval = ET.SubElement(lemma, 'eval')

  # read the input phonemes file and parse it into dictionary
  # output dictionary seq
  with codecs.open(bpe1k_file, 'rU', 'utf-8') as file:
    seq = {}
    for line in file:
      if line.startswith(('{', '}')):
        continue
      line = line.replace(',', '')
      line = line.replace('\'', '')
      key, value = line.strip().split(':')
      value = value.strip()
      seq[key] = value

  # create the xml file structure
  special_sign = ["L", "N", "S", "V"]
  extra_sign = ["</s>", "<s>", "<unk>"]

  # old lexicon handle
  lex = Lexicon(lexicon_file)

  count = 0
  temp_lemmas = []
  for word in lex.lemmas:
    count += 1
    if count > 9:
      if args.char:
        if hasNumber(lex.lemmas[word]['orth']):
          word_ = ""  
          list_ = separate(lex.lemmas[word]['orth'])
          for item in list_:
            if item.isdigit():
              word_ += convert(item)
          temp_lemmas.append(word_.strip())    
        else:
          temp_lemmas.append(lex.lemmas[word]['orth'])

  # create new lexicon root
  # create phonemes xml tree
  lex_root = ET.Element('lexicon')
  phone_inventory = ET.SubElement(lex_root, 'phoneme-inventory')
  for key, v in sorted(seq.items()):
    if key not in extra_sign:
      phone = ET.SubElement(phone_inventory, 'phoneme')
      p_sym = ET.SubElement(phone, 'symbol')
      p_var = ET.SubElement(phone, 'variation')

      if key in special_sign:
        p_var.text = 'none'
        if key == "L":
          p_sym.text = "[LAUGHTER]"
        elif key == "N":
          p_sym.text = "[NOISE]"
        elif key == "V":
          p_sym.text = "[VOCALIZEDNOISE]"
        else:
          p_sym.text = "[SILENCE]"
      else:
        p_var.text = 'context'
        p_sym.text = key
    else:
      if key == "<s>":
        create_specialTree(key)
      elif key == "</s>":
        create_specialTree(key)
      elif key == "<unk>":
        create_specialTree(key)

  for item in ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]:
    lemma = ET.SubElement(lex_root, 'lemma')
    orth = ET.SubElement(lemma, 'orth')
    phon = ET.SubElement(lemma, 'phon', score="0.0")
    phon.text = item
    orth.text = item
    synt = ET.SubElement(lemma, 'synt')
    eval = ET.SubElement(lemma, 'eval')

  # mapping phone sequences to word
  phon_dict = {}
    
  if args.char:
    for word in lex.lemmas:
      if hasNumber(word):
        word_ = ""  
        list_ = separate(word)
        for item in list_:
          if item.isdigit():
            word_ += convert(item)
        phon_dict[word] = word_
      else:
        phon_dict[word] = word
      #print(word, phon_dict[word])
  else:
    for word in lex.lemmas:  
      len_phons = len(lex.lemmas[word]["phons"])
      list_of_phons = []
      for x in range(len_phons):
        list_of_phons.append(lex.lemmas[word]["phons"][x]["phon"])
        if args.bpe:
          phon_dict[word] = to_unicode_list(list_of_phons) #phone bpe
        else:
          phon_dict[word] = list_of_phons #single phone

  if args.disamb:
    duplicates = {}  # phone -> count
    for word, phones in sorted(phon_dict.items()):
      for phone in phones:
        if phone in duplicates:
          phon_dict[word].remove(phone)
          phon_dict[word].insert(0, '%s #%s' % (phone, duplicates[phone])) #bpe close#, not bpe far #
          duplicates[phone] += 1
        else:
          duplicates[phone] = 1
    

  # auxiliary write a output file
  with open('word_phone.txt', 'w') as f:
    print(phon_dict, file=f)

  with open('file_to_map.txt', 'w') as file:
    file.write('{\n')
    for key, value in phon_dict.items():
      file.write('{}:{},\n'.format(key, value))
    file.write('}\n')

  with open('file_to_map.txt', 'r') as inp:
    with open('file_output.txt', 'w') as out:
      for i in range(6):
        inp.readline()
      for line in inp:
        if line.startswith('}'):
          break
        line = line.replace(',', '')
        _, right = line.split(':')
        lst = right[1:-2].split(',')
        lst = [x.replace("'", "") for x in lst]
        output = ' '.join(lst)
        out.write('{}\n'.format(output)) #for other add \n, without for SingleChar

  # here is the checkpoint, where ./subword-nmt/apply_bpe.py is called
  # with input files: codes file and phone sequences that to be map (e.g file_output.txt)
  # generate output: phones_bpe_file that will be used further

  with open(phones_bpe_file, 'r') as file_r:
    res_ = []
    for line in file_r:
      ls = line.strip().split()
      phon_seq = []
      merge = []
      for item in ls:
        if '@@' in item:
          merge.append(item)
        else:
          merge.append(item)
          phon_seq.append(' '.join(merge))
          merge = []
      res_.append(phon_seq)

  dict_tmp = list(phon_dict.items())

  for idx, x in enumerate(res_):
    dict_tmp[4+idx] = (dict_tmp[4+idx][0], x)

  phon_dict = dict(dict_tmp)

  with open('unicode_phone.txt', 'w') as f:
    print(phon_dict, file=f)

 # we want to add same words (ignoring case) to the same lemma so we create a dict from orth to
 # lemma to add a similar orth to the same lemma later. phon should be added only once to the lemma
 # so we do that when we create the lemma
  if args.char:
    orth_to_lemma = {}  # dict from orth to lemma
    for idx, elem in enumerate(temp_lemmas):
      elem_lower = elem.lower()
      # wenn schon drinne ist, gucken wir einfach nach
      if elem_lower in orth_to_lemma:
        lemma = orth_to_lemma[elem_lower]
      else:
        # wenn nicht, berechnet!
        lemma = ET.SubElement(lex_root, 'lemma')
        orth_to_lemma[elem_lower] = lemma
        #assert elem_lower in phon_dict
        res = ""
        for char in list(elem):
          res+=char
          res+=" "
        phon = ET.SubElement(lemma, 'phon')
        phon.text = res.strip()
      orth = ET.SubElement(lemma, 'orth')
      orth.text = elem
 
  # single char
#  if args.char:
#    orth_to_lemma = {}
#    for idx, elem in enumerate(temp_lemmas):
#      elem_lower = elem.lower()
#      lemma = ET.SubElement(lex_root, 'lemma')
#      orth = ET.SubElement(lemma, 'orth')
#      orth.text = elem
#      if elem_lower in orth_to_lemma:
#        lemma = orth_to_lemma[elem_lower]
#      else:
#        res = ""
#        for c in list(elem):
#          res+= c
#          res+= " "
#        phon = ET.SubElement(lemma, 'phon')
#        res = res + "<eow>"
#        phon.text = res
#  else:
#    orth_to_lemma = {}
#    for idx, elem in enumerate(temp_lemmas):
#      elem_lower = elem.lower()
#      lemma = ET.SubElement(lex_root, 'lemma')
#      orth = ET.SubElement(lemma, 'orth')
#      orth.text = elem
#      if elem_lower in orth_to_lemma:
#        lemma = orth_to_lemma[elem_lower]
#      else:
#        for p in phon_dict[elem_lower]:
#          phon = ET.SubElement(lemma, 'phon')
#          phon.text = p 

  else:
    orth_to_lemma = {}  # dict from orth to lemma
    for idx, elem in enumerate(temp_lemmas):
      elem_lower = elem.lower()
      # wenn schon drinne ist, gucken wir einfach nach
      if elem_lower in orth_to_lemma:
        lemma = orth_to_lemma[elem_lower]
      else:
        # wenn nicht, berechnet!
        lemma = ET.SubElement(lex_root, 'lemma')
        orth_to_lemma[elem_lower] = lemma
        assert elem_lower in phon_dict
        for p in phon_dict[elem_lower]:
          phon = ET.SubElement(lemma, 'phon')
          phon.text = p
      orth = ET.SubElement(lemma, 'orth')
      orth.text = elem

  if(args.output):
    my_data = minidom.parseString(ET.tostring(lex_root)).toprettyxml(indent="   ")
    with open(args.output, "w") as f:
      f.write(my_data)

if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  main()
