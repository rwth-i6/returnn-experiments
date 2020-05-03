from sisyphus import tk
from recipe.returnn import RETURNNTrainingFromFile
from recipe.returnn.search import RETURNNSearchFromFile, SearchBPEtoWords, ReturnnScore

def get_asr_dataset_stats(zip_dataset):
  """
  This function computes the global dataset statistics (mean and stddev) on a zip corpus to be used in the
  training dataset parameters of the OggZipDataset


  :param zip_dataset:
  :return:
  """

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
  """
  This function trains a RETURNN asr model, given the config and parameters

  :param config:
  :param name:
  :param parameter_dict:
  :return:
  """
  asr_train_job = RETURNNTrainingFromFile(config, parameter_dict=parameter_dict, mem_rqmt=16)
  asr_train_job.add_alias("asr_training/" + name)

  # asr_train_job.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

  asr_train_job.rqmt['time'] = 167
  asr_train_job.rqmt['cpu'] = 8
  tk.register_output("asr_training/" + name + "_model", asr_train_job.model_dir)
  tk.register_output("asr_training/" + name + "_training-scores", asr_train_job.learning_rates)
  return asr_train_job



def decode_and_evaluate_asr_config(name,
                                   config_file,
                                   model_path,
                                   epoch,
                                   zip_corpus,
                                   text,
                                   parameter_dict,
                                   training_name=None):
  """
  This function creates the RETURNN decoding/search job, converts the output into the format for scoring and computes
  the WER score

  :param str name: name of the decoding, usually the evaluation set name and decoding options
  :param Path config_file: training config or special decoding config file path
  :param Path model_path: .model_dir variable of the training job
  :param int|tk.Variable epoch: the epoch to select from the model folder
  :param Path zip_corpus: zip corpus for decoding
  :param Path text: text dictionary file for WER computation
  :param dict parameter_dict: network options
  :param str training_name: optional name of the trained model for alias and output naming
  :return:
  """

  with tk.block("evaluation"):
    path_prefix = "asr_evaluation/"
    if training_name:
      path_prefix += training_name + "/"

    local_parameter_dict = {'ext_eval_zip': zip_corpus,
                      'ext_decoding': True,
                      'ext_model': model_path,
                      'ext_load_epoch': epoch,
                     }
    if model_path == None:
      local_parameter_dict.pop("ext_model")
      local_parameter_dict.pop("ext_load_epoch")

    local_parameter_dict.update(parameter_dict)
    asr_recog_job = RETURNNSearchFromFile(config_file, parameter_dict=local_parameter_dict, mem_rqmt=12, time_rqmt=1,
                                   output_mode="py")

    # asr_recog_job.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

    asr_recog_job.add_alias(path_prefix + "search_%s/recognition" % name)
    tk.register_output(path_prefix + "search_%s/asr_out" % name, asr_recog_job.out)

    bpe_to_words_job = SearchBPEtoWords(asr_recog_job.out)
    bpe_to_words_job.add_alias(path_prefix + "search_%s/bpe_to_words" % name)
    tk.register_output(path_prefix + "search_%s/words_out" % name, bpe_to_words_job.out)

    wer_score_job = ReturnnScore(bpe_to_words_job.out, text)
    wer_score_job.add_alias(path_prefix + "search_%s/wer_scoring" % name)
    tk.register_output(path_prefix + "search_%s/WER" % name, wer_score_job.out)

    return wer_score_job.out
