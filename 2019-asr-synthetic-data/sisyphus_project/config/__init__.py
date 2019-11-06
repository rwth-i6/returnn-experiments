from sisyphus import *

Path = setup_path(__package__)

from recipe.corpus.librispeech import LibriSpeechToBliss
from recipe.corpus import BlissToZipDataset
from recipe.text.bliss import BlissExtractRawText
from recipe.text.subword_units import CreateSubwordsAndVocab

def prepare_data():

  dataset_names = ['dev-clean', 'dev-other', 'test-clean', 'test-other',
                   'train-clean-100', 'train-clean-360']

  bliss_flac_corpus_dict = {}
  zip_flac_corpus_dict = {}

  for dataset_name in dataset_names:
    dataset_path = Path("../data/dataset-raw/LibriSpeech/%s/" % dataset_name)

    ls_to_bliss_job = LibriSpeechToBliss(corpus_path=dataset_path, name=dataset_name)
    ls_to_bliss_job.add_alias("data/LibriSpeechToBliss/%s" % dataset_name)
    bliss_flac_corpus_dict[dataset_name] = ls_to_bliss_job.out
    tk.register_output("data/bliss/%s.xml.gz" % dataset_name, ls_to_bliss_job.out)

    bliss_to_zip_job = BlissToZipDataset(name=dataset_name, corpus_file=ls_to_bliss_job.out, use_full_seq_name=False)
    bliss_to_zip_job.add_alias("data/BlissToZipDataset/%s" % dataset_name)
    zip_flac_corpus_dict[dataset_name] = bliss_to_zip_job.out
    tk.register_output("data/asr_zip/%s.zip" % dataset_name, bliss_to_zip_job.out)

  return bliss_flac_corpus_dict, zip_flac_corpus_dict


def build_subwords(bliss_corpora):
  """

  :param list bliss_corpora:
  :return:
  """
  corpus_texts = []
  for bliss_corpus in bliss_corpora:
    extract_text_job = BlissExtractRawText(bliss_corpus)
    corpus_texts.append(extract_text_job.out)

  from recipe.text import Concatenate
  text = Concatenate(corpus_texts).out
  subwords_job = CreateSubwordsAndVocab(text=text, num_segments=10000)
  subwords_job.add_alias("data/subwords/CreateSubwordsAndVocab")

  bpe_codes = subwords_job.out_bpe
  bpe_vocab = subwords_job.out_vocab
  bpe_vocab_size = subwords_job.out_vocab_size

  tk.register_output("data/subwords/bpe.codes", bpe_codes)
  tk.register_output("data/subwords/bpe.vocab", bpe_vocab)
  tk.register_output("data/subwords/bpe.vocab_size", bpe_vocab_size)

  return bpe_codes, bpe_vocab, bpe_vocab_size


def get_asr_dataset_stats(zip_dataset):

  config = {'train':
              {'class': 'OggZipDataset',
               'audio': {},
               'targets': None,
               'path': zip_dataset}
           }

  from recipe.returnn.dataset import ExtractDatasetStats
  dataset_stats_job = ExtractDatasetStats(config)
  dataset_stats_job.add_alias("data/stats/ExtractDatasetStats")

  mean = dataset_stats_job.mean_file
  std_dev = dataset_stats_job.std_dev_file

  tk.register_output('data/stats/norm.mean.txt', mean)
  tk.register_output('data/stats/norm.std_dev.txt', std_dev)

  return mean, std_dev

def train_asr_config(config, name, parameter_dict=None):
  from recipe.returnn import RETURNNTrainingFromFile
  asr_train = RETURNNTrainingFromFile(config, parameter_dict=parameter_dict, mem_rqmt=16)
  asr_train.add_alias("asr_training/" + name)

  # TODO: Remove
  asr_train.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

  asr_train.rqmt['time'] = 167
  asr_train.rqmt['cpu'] = 8
  tk.register_output("asr_training/" + name + "_model", asr_train.model_dir)
  tk.register_output("asr_training/" + name + "_training-scores", asr_train.learning_rates)
  return asr_train

def main():
  bliss_dict, zip_dict = prepare_data()

  bpe_codes, bpe_vocab, num_classes = build_subwords([bliss_dict['train-clean-100'],
                                                      bliss_dict['train-clean-360']])

  mean, stddev = get_asr_dataset_stats(zip_dict['train-clean-100'])

  asr_global_parameter_dict = {
    'ext_norm_mean': mean,
    'ext_norm_std_dev': stddev,
    'ext_bpe_file': bpe_codes,
    'ext_vocab_file': bpe_vocab,
    'ext_num_classes': num_classes
  }


  initial_checkpoint_training_params = {
    'ext_partition_epoch': 20,
    'ext_training_zips': [zip_dict['train-clean-100']],
    'ext_dev_zips': [zip_dict['dev-clean'],
                     zip_dict['dev-other']],
    'ext_num_epochs': 80
  }

  initial_checkpoint_training_params.update(asr_global_parameter_dict)

  asr_training_config = Path("returnn_configs/asr/train-clean-100.exp3.ctc.ogg.lrwarmupextra10.config")

  initial_training_job = train_asr_config(asr_training_config, "librispeech-100-initial-training",
                             initial_checkpoint_training_params)

