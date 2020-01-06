from recipe.corpus import BlissToZipDataset
from recipe.corpus.librispeech import LibriSpeechToBliss
from recipe.text.bliss import BlissExtractRawText, BlissExtractTextDictionary
from recipe.text.subword_units import CreateSubwordsAndVocab
from sisyphus import tk, setup_path

Path = setup_path(__package__)

def prepare_data_librispeech():
  """
  This function creates the LibriSpeech data in Bliss format and zip format.
  For the evaluation sets, the text is extracted in dictionary form for WER scoring

  :return:
  """

  # all datasets that are used in the experiments for LibriSpeech
  dataset_names = ['dev-clean', 'dev-other', 'test-clean', 'test-other',
                   'train-clean-100', 'train-clean-360']

  evaluation_names = ['dev-clean', 'dev-other', 'test-clean', 'test-other']


  bliss_flac_corpus_dict = {}
  zip_flac_corpus_dict = {}
  transcription_corpus_dict = {}

  for dataset_name in dataset_names:
    dataset_path = Path("../data/dataset-raw/LibriSpeech/%s/" % dataset_name)

    # open the raw LibriSpeech data and create bliss corpus
    ls_to_bliss_job = LibriSpeechToBliss(corpus_path=dataset_path, name=dataset_name)
    ls_to_bliss_job.add_alias("data/LibriSpeechToBliss/%s" % dataset_name)
    bliss_flac_corpus_dict[dataset_name] = ls_to_bliss_job.out
    tk.register_output("data/bliss/%s.xml.gz" % dataset_name, ls_to_bliss_job.out)

    # create a unified zip corpus file from the bliss corpus
    bliss_to_zip_job = BlissToZipDataset(name=dataset_name, corpus_file=ls_to_bliss_job.out, use_full_seq_name=False)
    bliss_to_zip_job.add_alias("data/BlissToZipDataset/%s" % dataset_name)
    zip_flac_corpus_dict[dataset_name] = bliss_to_zip_job.out
    tk.register_output("data/asr_zip/%s.zip" % dataset_name, bliss_to_zip_job.out)

  for dataset_name in evaluation_names:
    # create the dictionary format transcription files
    bliss_to_text_dict_job = BlissExtractTextDictionary(bliss_flac_corpus_dict[dataset_name], segment_key_only=True)
    bliss_to_text_dict_job.add_alias("data/BlissExtractTextDictionary/%s" % dataset_name)
    transcription_corpus_dict[dataset_name] = bliss_to_text_dict_job.out

  return bliss_flac_corpus_dict, zip_flac_corpus_dict, transcription_corpus_dict


def build_subwords(bliss_corpora, num_segments, name):
  """
  This function creates the subword codes and vocabulary files for a given bliss dataset

  :param list bliss_corpora: bliss corpus for subword training
  :param int num_segments: number of bpe merge operations / bpe segments
  :param str name: name of the subwords
  :return:
  """
  corpus_texts = []
  for bliss_corpus in bliss_corpora:
    extract_text_job = BlissExtractRawText(bliss_corpus)
    corpus_texts.append(extract_text_job.out)

  from recipe.text import Concatenate
  text = Concatenate(corpus_texts).out
  subwords_job = CreateSubwordsAndVocab(text=text, num_segments=num_segments)
  subwords_job.add_alias("data/subwords/CreateSubwordsAndVocab-%s" % name)

  bpe_codes = subwords_job.out_bpe
  bpe_vocab = subwords_job.out_vocab
  bpe_vocab_size = subwords_job.out_vocab_size

  tk.register_output("data/subwords/%s.bpe.codes" % name, bpe_codes)
  tk.register_output("data/subwords/%s.bpe.vocab" % name, bpe_vocab)
  tk.register_output("data/subwords/%s.bpe.vocab_size" % name, bpe_vocab_size)

  return bpe_codes, bpe_vocab, bpe_vocab_size