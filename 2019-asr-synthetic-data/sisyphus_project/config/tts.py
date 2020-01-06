from recipe.corpus import BlissToZipDataset
from recipe.default_values import FFMPEG_BINARY
from sisyphus import tk


def process_corpus(bliss_corpus, char_vocab, silence_duration, name=None):
  """
  process a bliss corpus to be suited for TTS training
  :param self:
  :param bliss_corpus:
  :param name:
  :return:
  """
  from recipe.text.bliss import ProcessBlissText
  ljs = ProcessBlissText(bliss_corpus, [('end_token',{'token': '~'})],
                         vocabulary=char_vocab)

  from recipe.corpus.ffmpeg import BlissFFMPEGJob, BlissRecoverDuration

  filter_string = '-af "silenceremove=stop_periods=-1:window=%f:stop_duration=0.01:stop_threshold=-40dB"' % \
                  silence_duration

  ljs_nosilence = BlissFFMPEGJob(ljs.out, filter_string, ffmpeg_binary=FFMPEG_BINARY, output_format="wav")
  ljs_nosilence.rqmt['time'] = 24

  ljs_nosilence_recover = BlissRecoverDuration(ljs_nosilence.out)

  return ljs_nosilence_recover.out


def prepare_tts_data(bliss_dict):
  """

  :param dict bliss_dict:
  :return:
  """

  from recipe.returnn.vocabulary import BuildCharacterVocabulary
  build_char_vocab_job = BuildCharacterVocabulary(uppercase=True)
  char_vocab = build_char_vocab_job.out

  processed_corpora = {}
  processed_zip_corpora = {}
  for name, corpus in bliss_dict.items():
    tts_name = "tts-" + name
    processed_corpus = process_corpus(bliss_corpus=corpus,
                                      char_vocab=char_vocab,
                                      silence_duration=0.1,
                                      name=tts_name)
    processed_corpora[tts_name] = processed_corpus
    tk.register_output("data/bliss/%s.processed.xml.gz" % name, processed_corpus)

    processed_zip_corpora[tts_name] = BlissToZipDataset(tts_name, processed_corpus).out

  return processed_corpora, processed_zip_corpora, char_vocab


def get_tts_dataset_stats(zip_dataset):

  config = {'train':
              {'class': 'OggZipDataset',
               'audio': {'feature_options': {'fmin': 60},
                         'features': 'db_mel_filterbank',
                         'num_feature_filters': 80,
                         'peak_normalization': False,
                         'preemphasis': 0.97,
                         'step_len': 0.0125,
                         'window_len': 0.05},
               'targets': None,
               'path': zip_dataset}
            }

  from recipe.returnn.dataset import ExtractDatasetStats
  dataset_stats_job = ExtractDatasetStats(config)
  dataset_stats_job.add_alias("data/tts_stats/ExtractDatasetStats")

  mean = dataset_stats_job.mean
  std_dev = dataset_stats_job.std_dev

  tk.register_output('data/tts_stats/norm.mean.txt', mean)
  tk.register_output('data/tts_stats/norm.std_dev.txt', std_dev)

  return mean, std_dev


def train_tts_config(config, name, parameter_dict=None):
  from recipe.returnn import RETURNNTrainingFromFile
  asr_train = RETURNNTrainingFromFile(config, parameter_dict=parameter_dict, mem_rqmt=16)
  asr_train.add_alias("tts_training/" + name)

  # TODO: Remove
  asr_train.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

  asr_train.rqmt['time'] = 167
  asr_train.rqmt['cpu'] = 8
  tk.register_output("tts_training/" + name + "_model", asr_train.model_dir)
  tk.register_output("tts_training/" + name + "_training-scores", asr_train.learning_rates)
  return asr_train