from recipe.corpus import BlissToZipDataset, BlissAddTextFromBliss
from recipe.default_values import FFMPEG_BINARY
from sisyphus import tk, Path

from config.f2l import convert_with_f2l, griffin_lim_ogg


def process_corpus(bliss_corpus, char_vocab, silence_duration):
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

  filter_string = '-af "silenceremove=stop_periods=-1:stop_duration=%f:stop_threshold=-40dB"' % \
                  silence_duration

  ljs_nosilence = BlissFFMPEGJob(ljs.out, filter_string, ffmpeg_binary=FFMPEG_BINARY, output_format="wav")
  ljs_nosilence.rqmt['time'] = 24

  ljs_nosilence_recover = BlissRecoverDuration(ljs_nosilence.out)

  return ljs_nosilence_recover.out


def prepare_ttf_data(bliss_dict):
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
                                      silence_duration=0.1)
    processed_corpora[tts_name] = processed_corpus
    tk.register_output("data/bliss/%s.processed.xml.gz" % name, processed_corpus)

    processed_zip_corpora[tts_name] = BlissToZipDataset(tts_name, processed_corpus).out

  return processed_corpora, processed_zip_corpora, char_vocab


def get_ttf_dataset_stats(zip_dataset):

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


def train_ttf_config(config, name, parameter_dict=None):
  from recipe.returnn import RETURNNTrainingFromFile
  asr_train = RETURNNTrainingFromFile(config, parameter_dict=parameter_dict, mem_rqmt=16)
  asr_train.add_alias("tts_training/" + name)

  # asr_train.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

  asr_train.rqmt['time'] = 167
  asr_train.rqmt['cpu'] = 8
  tk.register_output("tts_training/" + name + "_model", asr_train.model_dir)
  tk.register_output("tts_training/" + name + "_training-scores", asr_train.learning_rates)
  return asr_train

def generate_speaker_embeddings(config_file, model_dir, epoch, zip_corpus, name, default_parameter_dict=None):
  from recipe.returnn.forward import RETURNNForwardFromFile


  parameter_dict = {'ext_gen_speakers': True,
                    'ext_model': model_dir,
                    'ext_load_epoch': epoch,
                    'ext_eval_zip': zip_corpus}

  parameter_dict.update(default_parameter_dict)

  generate_speaker_embeddings_job = RETURNNForwardFromFile(config_file,
                                                           parameter_dict=parameter_dict,
                                                           hdf_outputs=['speaker_embeddings'],
                                                           mem_rqmt=8)
  generate_speaker_embeddings_job.add_alias("tts_speaker_generation/" + name)
  tk.register_output("tts_speaker_generation/" + name + "_speakers.hdf",
                     generate_speaker_embeddings_job.outputs['speaker_embeddings'])

  return generate_speaker_embeddings_job.outputs['speaker_embeddings']

def decode_with_speaker_embeddings(config_file, model_dir, epoch, zip_corpus, speaker_hdf, name,
                                   default_parameter_dict=None, segment_file=None):
  from recipe.returnn.forward import RETURNNForwardFromFile
  from recipe.tts.toolchain import ConvertFeatures

  parameter_dict = {'ext_forward': True,
                    'ext_model': model_dir,
                    'ext_load_epoch': epoch,
                    'ext_eval_zip': zip_corpus,
                    'ext_speaker_embeddings': speaker_hdf,
                    'ext_forward_segment_file': segment_file}

  if segment_file == None:
      parameter_dict.pop('ext_forward_segment_file')

  parameter_dict.update(default_parameter_dict)

  decode_with_speakers_job = RETURNNForwardFromFile(config_file,
                                                           parameter_dict=parameter_dict,
                                                           hdf_outputs=['stacked_features'],
                                                           mem_rqmt=8)

  decode_with_speakers_job.rqmt['qsub_args'] = '-l h_fsize=200G'

  decode_with_speakers_job.add_alias("tts_decode_with_speakers/" + name)

  convert_features_job = ConvertFeatures(decode_with_speakers_job.outputs['stacked_features'], 3)
  convert_features_job.add_alias("tts_convert_features/" + name)

  return convert_features_job.out, decode_with_speakers_job, convert_features_job



def evaluate_tts(ttf_config_file, ttf_model_dir, ttf_epoch,
                 f2l_config_file, f2l_model_dir, f2l_epoch,
                 train_zip_corpus, test_zip_corpus, test_bliss_corpus, test_text,
                 default_parameter_dict, name):

    embedding_hdf = generate_speaker_embeddings(ttf_config_file, ttf_model_dir, ttf_epoch, train_zip_corpus,
                                             name=name, default_parameter_dict=default_parameter_dict)

    from recipe.tts.corpus import DistributeSpeakerEmbeddings
    dist_speaker_embeds_job = DistributeSpeakerEmbeddings(test_bliss_corpus, embedding_hdf,
                                                          use_full_seq_name=False, options=None)
    dist_embeddings_hdf = dist_speaker_embeds_job.out

    unstacked_feature_hdf, _, _ =  decode_with_speaker_embeddings(ttf_config_file, ttf_model_dir, ttf_epoch, test_zip_corpus,
                                   dist_embeddings_hdf, name, default_parameter_dict)

    linear_features, _ = convert_with_f2l(f2l_config_file, name, f2l_model_dir, f2l_epoch, unstacked_feature_hdf)

    generated_audio_bliss, _ = griffin_lim_ogg(linear_features, name)

    asr_bliss = BlissAddTextFromBliss(generated_audio_bliss, test_bliss_corpus).out

    from recipe.corpus import BlissToZipDataset
    asr_zip = BlissToZipDataset("test", asr_bliss, use_full_seq_name=False).out

    from config.asr import decode_and_evaluate_asr_config
    # ASR EVALUATION
    trafo_specaug = Path(
        "/u/rossenbach/experiments/switchboard_test/config/alberts_configs/trafo.specaug4.12l.ffdim4.pretrain3.natctc_recognize_pretrained.config")

    decode_and_evaluate_asr_config("test-%s" % name, trafo_specaug, None, 0, asr_zip, test_text, {})
