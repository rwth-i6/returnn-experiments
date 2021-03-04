#!crnn/rnn.py
# kate: syntax python;
# vim: ft=python sw=2:
# based on Andre Merboldt rnnt-fs.bpe1k.readout.zoneout.lm-embed256.lr1e_3.no-curric.bs12k.mgpu.retrain1.config

import os
from returnn.tf.util.data import DimensionTag, Data
from returnn.import_ import import_

import_("github.com/rwth-i6/returnn-experiments", "common")
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.datasets.asr.librispeech import oggzip
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.transducer_fullsum import make_net

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
globals().update(oggzip.Librispeech().get_config_opts())

network = make_net(task=task, target=globals()["target"])
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


num_epochs = 100
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
newbob_multi_num_epochs = globals().get("train", {}).get("partition_epoch", 1)
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.9
learning_rate_file = "newbob.data"

# log
# log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
model_name = os.path.splitext(os.path.basename(__file__))[0]
log = "/var/tmp/am540506/log/%s/crnn.%s.log" % (model_name, task)
log_verbosity = 5
os.makedirs(os.path.dirname(log), exist_ok=True)

