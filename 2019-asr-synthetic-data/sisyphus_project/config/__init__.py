from config.asr import get_asr_dataset_stats, train_asr_config, decode_and_evaluate_asr_config
from config.data import prepare_data_librispeech, build_subwords
from config.tts import prepare_tts_data, get_tts_dataset_stats, train_tts_config
from sisyphus import *
import copy

Path = setup_path(__package__)


def main():
  # prepare the datasets in bliss and zip format
  bliss_dict, zip_dict, transcription_text_dict = prepare_data_librispeech()

  # compute the subword codes and the ASR vocabulary
  bpe_codes, bpe_vocab, num_classes = build_subwords([bliss_dict['train-clean-100'],
                                                      bliss_dict['train-clean-360']], num_segments=10000,
                                                     name="librispeech-460")

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

  # training of the initial model
  initial_checkpoint_training_params.update(asr_global_parameter_dict)

  asr_training_config = Path("returnn_configs/asr/train-clean-100.exp3.ctc.ogg.lrwarmupextra10.config")
  initial_training_job = train_asr_config(asr_training_config, "librispeech-100-initial-training",
                                          initial_checkpoint_training_params)

  # training and decoding of the baseline model
  baseline_training_params = copy.deepcopy(initial_checkpoint_training_params)
  baseline_training_params['ext_num_epochs'] = 170

  baseline_training_params['import_model_train_epoch1'] = initial_training_job.models[80].model
  baseline_training_params.update(asr_global_parameter_dict)
  continued_training_job = train_asr_config(asr_training_config, "librispeech-100-baseline-training",
                                            baseline_training_params)

  from recipe.returnn.search import GetBestEpoch
  best_epoch = GetBestEpoch(continued_training_job.model_dir, continued_training_job.learning_rates,
                            key="dev_score_output/output_prob").out_var
  tk.register_output("test_best_epoch", best_epoch)

  for key in transcription_text_dict:
    wer = decode_and_evaluate_asr_config(key,
                                         asr_training_config,
                                         continued_training_job.model_dir,
                                         epoch=best_epoch,
                                         zip_corpus=zip_dict[key],
                                         text=transcription_text_dict[key],
                                         parameter_dict=asr_global_parameter_dict,
                                         training_name="baseline")


  ##########################3
  # TTS

  tts_bliss_dict = {k:v for k,v in bliss_dict.items() if k in ['dev-clean', 'train-clean-100']}
  tts_bliss_corpora, tts_zip_corpora, char_vocab = prepare_tts_data(tts_bliss_dict)

  mean, stddev = get_tts_dataset_stats(tts_zip_corpora['tts-train-clean-100'])

  tts_global_parameter_dict = {
    'ext_norm_mean_value': mean,
    'ext_norm_std_dev_value': stddev,
    'ext_char_vocab': char_vocab,
    'ext_training_zips': [tts_zip_corpora['tts-train-clean-100']],
    'ext_dev_zips': [tts_zip_corpora['tts-dev-clean']],
    'ext_num_epochs': 200,
    'ext_partition_epoch': 20,
  }

  tts_training_config = Path("returnn_configs/tts/tts-clean-100.dec640.enc256.enclstm512.config")
  tts_training_job = train_tts_config(tts_training_config, name="tts-baseline-training",
                                      parameter_dict=tts_global_parameter_dict)
