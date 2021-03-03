#!crnn/rnn.py
# kate: syntax python;
# vim: ft=python sw=2:
# based on Andre Merboldt rnnt-fs.bpe1k.readout.zoneout.lm-embed256.lr1e_3.no-curric.bs12k.mgpu.retrain1.config

import os
from returnn.tf.util.data import DimensionTag, Data
from returnn.import_ import import_

import_("github.com/rwth-i6/returnn-experiments", "common")
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.asr.specaugment import specaugment
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.recomb_recog import targetb_recomb_recog
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.loss import rnnt_loss
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.collect_out_str import out_str

config = globals()["config"]  # make PyCharm happy

use_horovod = config.bool("use_horovod", False)
horovod_dataset_distribution = "random_seed_offset"
horovod_reduce_type = "param"
#horovod_param_sync_step = 100
horovod_param_sync_time_diff = 100.
# horovod_scale_lr = True

if use_horovod:
  import socket
  prefix = "%s-pid%i:" % (socket.gethostname(), os.getpid())
  print(prefix, "use_horovod, CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
  import TFHorovod
  # Important: Horovod options must be defined before this call!
  hvd = TFHorovod.get_ctx(config=config)
  print(prefix, "Local rank/size:", hvd.local_rank(), hvd.local_size())

# Workaround for openblas hanging:
# * https://github.com/tensorflow/tensorflow/issues/13802
# * https://github.com/rwth-i6/returnn/issues/323#issuecomment-725384762
patch_atfork = not use_horovod

# task
use_tensorflow = True
task = config.value("task", "train")
device = "gpu"
multiprocessing = True
update_on_device = True

debug_mode = False
if int(os.environ.get("RETURNN_DEBUG", "0")):
  import sys
  print("** DEBUG MODE", file=sys.stderr)
  debug_mode = True

if config.has("beam_size"):
  beam_size = config.int("beam_size", 0)
  import sys
  print("** beam_size %i" % beam_size, file=sys.stderr)
else:
  beam_size = 12

# data
_time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
_target = "classes"
_target_num_labels = 1056
_targetb_num_labels = _target_num_labels + 1
_targetb_blank_idx = _target_num_labels

extern_data = {
    _target: {"dim": _target_num_labels, "sparse": True},  # see vocab
    "data": {"dim": 40, "same_dim_tags_as": {"t": _time_tag}},  # Gammatone 40-dim
    }
if task != "train":
  extern_data["targetb"] = {"dim": _targetb_num_labels, "sparse": True, "available_for_inference": False}

_epoch_split = 20


def get_dataset(key, subset=None, train_partition_epoch=None):
  dataset_dir = "/var/tmp/am540506/librispeech/dataset"
  d = {
    'class': 'LibriSpeechCorpus',
    'path': dataset_dir,
    "use_zip": True,
    "use_ogg": False,
    "use_cache_manager": not debug_mode,
    "prefix": key,
    "bpe": {
      'bpe_file': '%s/trans.bpe_1000.codes' % dataset_dir,
      'vocab_file': '%s/trans.bpe_1000.vocab' % dataset_dir,
      'unknown_label': '<unk>'},
    "audio": {
      "norm_mean": "base/dataset/stats.mean.txt",
      "norm_std_dev": "base/dataset/stats.std_dev.txt"},
  }
  if key.startswith("train"):
    d["partition_epoch"] = train_partition_epoch
    if key == "train":
      d["epoch_wise_filter"] = {
        (1, 20): {
          'use_new_filter': True,
          'subdirs': ['train-clean-100', 'train-clean-360']},
        }
    #d["audio"]["random_permute"] = True
    num_seqs = 281241  # total
    d["seq_ordering"] = "laplace:%i" % (num_seqs // 1000)
  else:
    d["fixed_random_seed"] = 1
    d["seq_ordering"] = "sorted_reverse"
  if subset:
    d["fixed_random_subset"] = subset  # faster
  return d


train = get_dataset("train", train_partition_epoch=_epoch_split)
dev = get_dataset("dev", subset=3000)
eval_datasets = {"devtrain": get_dataset("train", subset=2000)}
cache_size = "0"
window = 1



search_output_layer = "decision"
debug_print_layer_output_template = True

# trainer
batching = "random"
log_batch_size = True
batch_size = 12000
max_seqs = 200
max_seq_length = {"classes": 75}
#chunking = ""  # no chunking
truncation = -1


num_epochs = _range_epochs_full_sum [1] + 1
model = "net-model/network"
cleanup_old_models = True
gradient_clip = 0
#gradient_clip_global_norm = 1.0

adam = True
optimizer_epsilon = 1e-8
#debug_add_check_numerics_ops = True
#debug_add_check_numerics_on_output = True
stop_on_nonfinite_train_score = False
tf_log_memory_usage = True
gradient_noise = 0.0
learning_rate = 0.001
learning_rate_control = "newbob_multi_epoch"
#learning_rate_control_error_measure = "dev_score_output"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
use_learning_rate_control_always = True
newbob_multi_num_epochs = _epoch_split
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.9
learning_rate_file = "newbob.data"

# log
#log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
model_name = os.path.splitext(os.path.basename(__file__))[0]
log = "/var/tmp/am540506/log/%s/crnn.%s.log" % (model_name, task)
log_verbosity = 5
os.makedirs(os.path.dirname(log), exist_ok=True)

