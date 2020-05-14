
from returnn.LmDataset import Lexicon

def save_corpus_segments_to_file(output_filename, input_dict):
  """
  :param str output_filename:
  :param dict[str,str] input_dict:
  """
  # write result into scoring-dev-phones.words
  with open(output_filename, 'w') as file:
    file.write('{\n')
    for key, value in input_dict.items():
      value = value.lstrip()
      file.write('\"{}\": \"{}\",\n'.format(key, value))
    file.write('}\n')


class PhoneMapper:
  def __init__(self, lexicon_filename, phone_unicode_map_filename=None):
    """
    :param str lexicon_filename:
    :param str phone_unicode_map_filename:
    """
    self._lexicon_filename = lexicon_filename
    self._phone_unicode_map = phone_unicode_map_filename
    self._lex_dict = {}  # word -> list of phone-str
    self.lexicon = None
    self.phone_unicode_map = None
    self._loaded = False

  def _lazy_load(self):
    if self._loaded:
      return

    self._loaded = True
    self.lexicon = Lexicon(self._lexicon_filename)
    if self._phone_unicode_map:
        self.phone_unicode_map = eval(open(self._phone_unicode_map).read())  # phone -> single-unicode-char

    for word in self.lexicon.lemmas:
      list_phones = []
      for item in self.lexicon.lemmas[word]['phons']:
        list_phones.append(item['phon'])
      if self.phone_unicode_map:
        self._lex_dict[word] = self._to_unicode_list(list_phones)
      else:
        self._lex_dict[word] = list_phones

    duplicates = {}  # phone -> count
    for word, phones in sorted(self._lex_dict.items()):
      for phone in phones:
        if phone in duplicates:
          self._lex_dict[word].remove(phone)
          self._lex_dict[word].insert(0, '%s#%s' % (phone, duplicates[phone]))
          duplicates[phone] += 1
        else:
          duplicates[phone] = 1

  def map_words_to_phones(self, seq):
    """
    :param str seq: white-space separated string, seq of words
    :rtype: str
    """
    self._lazy_load()
    words = seq.split()
    result = []
    for word in words:
      if word in self._lex_dict:
        result.append("".join((self._lex_dict[word][0]).split()))
    return " ".join(result)

  def _to_unicode(self, phones):
    """
    :param str phones:
    :rtype: str
    """
    phone_list = phones.split()
    result = ""
    for k in phone_list:
      result += self.phone_unicode_map[k]
    return result

  def _to_unicode_list(self, list_of_phones):
    """
    :param list[str] list_of_phones:
    :rtype: list[str]
    """
    res = []
    for item in list_of_phones:
      res.append(self._to_unicode(item))
    return res

  def get_phones_to_word_mapping(self, disamb=True):
    """
    :return: phones->word
    :rtype: dict[str,str]
    """
    self._lazy_load()
    reverse_lex_dict = {}
    for word, phone_list in self._lex_dict.items():
      for phone_seq in phone_list:
        reverse_lex_dict[phone_seq] = word
    return reverse_lex_dict

  def map_phones_to_word_on_corpus(self, segments):
    """
    :param dict[str,str] segments: seq_tag->seq of unicode phones
    :return: seq_tag->seq of words
    :rtype: dict[str,str]
    """
    reverse_lex_dict = self.get_phones_to_word_mapping()
    result_disamb = {}  # seq_tag -> seq of words
    for seg_tag, word_seq in segments.items():
      transform = []
      for unicode in word_seq.split():
        if unicode in reverse_lex_dict:
          transform.append(reverse_lex_dict[unicode])
        else:
          transform.append("[UNKNOWN]")
      result_disamb[seg_tag] = " ".join(transform)
    return result_disamb
