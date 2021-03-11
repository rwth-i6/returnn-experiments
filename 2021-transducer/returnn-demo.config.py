#!crnn/rnn.py
# kate: syntax python;
# vim: ft=python sw=2:
# based on Andre Merboldt rnnt-fs.bpe1k.readout.zoneout.lm-embed256.lr1e_3.no-curric.bs12k.mgpu.retrain1.config

from returnn.import_ import import_

import_("github.com/rwth-i6/returnn-experiments", "common")
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.common_config import *
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.datasets.asr.timit.nltk import NltkTimit
from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common.models.transducer.transducer_fullsum import make_net

# data
globals().update(NltkTimit().get_config_opts())

network = make_net(
  task=task,
  encoder_opts=dict(num_layers=3, lstm_dim=128, dropout=0.05),
  decoder_opts=dict(
    lm_embed_dim=64,
    lm_dropout=0,
    lm_lstm_dim=128,
    readout_dropout=0.1,
    readout_dim=256,
    output_dropout=0.05,
  ))

# trainer
batching = "random"
batch_size = 1000
max_seqs = 10 if debug_mode else 200
max_seq_length = {"classes": 75}

num_epochs = 100
model = "net-model/network"
cleanup_old_models = True

adam = True
optimizer_epsilon = 1e-8
# debug_add_check_numerics_ops = True
# debug_add_check_numerics_on_output = True
stop_on_nonfinite_train_score = False
gradient_noise = 0.0
gradient_clip = 0
# gradient_clip_global_norm = 1.0

learning_rate = 0.001
learning_rate_control = "newbob_multi_epoch"
# learning_rate_control_error_measure = "dev_score_output"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
use_learning_rate_control_always = True
newbob_multi_num_epochs = globals().get("train", {}).get("partition_epoch", 1)
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.9
learning_rate_file = "newbob.data"

# log
# log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
# model_name = os.path.splitext(os.path.basename(__file__))[0]
# log = "/var/tmp/am540506/log/%s/crnn.%s.log" % (model_name, task)
# os.makedirs(os.path.dirname(log), exist_ok=True)
log_verbosity = 5
