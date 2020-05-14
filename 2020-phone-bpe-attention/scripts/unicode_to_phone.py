from returnn.LmDataset import Lexicon
from argparse import ArgumentParser
import PhoneUtils

def parse_vocab(filename):
  """
  :param str filename:
  :rtype: dict[str,str]
  :return: phone->unicode
  """
  raw = open(filename, "r").read()
  return eval(raw)

def main():
  args_parser = ArgumentParser()
  args_parser.add_argument("--lexicon", required=True)
  args_parser.add_argument("--input", required=True)  # scoring-dev.words
  args_parser.add_argument("--disamb_map", required=True)
  args_parser.add_argument("--disamb", action="store_true")
  args_parser.add_argument("--output", required=True)
  args = args_parser.parse_args()

  lexicon_fn = args.lexicon
  disamb_map_fn = args.disamb_map
  scoring_words = args.input

  map_dict = parse_vocab(disamb_map_fn)  # read map dict
  segments = eval(open(scoring_words, 'r').read())  # read word sequences
  unicode_to_phone = {value: key for key, value in map_dict.items()}

  phone_mapper = PhoneUtils.PhoneMapper(lexicon_filename=lexicon_fn, phone_unicode_map_filename=disamb_map_fn)

  if args.disamb:
    result_disamb = phone_mapper.map_phones_to_word_on_corpus(segments)
    PhoneUtils.save_corpus_segments_to_file(args.output, result_disamb)
  else:
    # map the unicode to phone
    # map shouldn't handle disambiguity symbol #1,#2,..etc
    result = {}
    for seg_tag, word_seq in segments.items():
      phones_seq = ''
      for unicode in word_seq:
        phones_seq += unicode_to_phone[unicode]
      result[seg_tag] = phones_seq

    # map phones sequences to words using lexicon
    lex = Lexicon(lexicon_fn)
    vocab_dict = {}
    words_map = {}

    for word in lex.lemmas:
      if len(lex.lemmas[word]["phons"]) > 0:
        phonemes = lex.lemmas[word]["phons"][0]["phon"]
        phone_concat = "".join(phonemes.split())  # phone_concat -> list of possible words
        vocab_dict[word] = phone_concat
        words_map.setdefault(phone_concat, []).append(word)

    for k, v in result.items():
      list_phones = v.split()
      new_value = ""
      for phones in list_phones:
        if phones in words_map.keys():
          new_value += ' ' + words_map[phones][0]
        else:
          new_value += ' ' + '[UNK]'
      result[k] = new_value

    PhoneUtils.save_corpus_segments_to_file(args.output, result)


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  main()
