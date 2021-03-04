
"""
Provides some common settings for Horovod multi-GPU training
(but only used if ``use_horovod`` was set earlier in the config, or externally).

Use `from ... import *` directly in your config for this module.
"""

import os
from returnn.config import get_global_config

config = get_global_config()

use_horovod = config.bool("use_horovod", False)
horovod_dataset_distribution = "random_seed_offset"
horovod_reduce_type = "param"
# horovod_param_sync_step = 100
horovod_param_sync_time_diff = 100.
# horovod_scale_lr = True

if use_horovod:
  import socket
  prefix = "%s-pid%i:" % (socket.gethostname(), os.getpid())
  print(prefix, "use_horovod, CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
  import returnn.tf.horovod
  # Important: Horovod options must be defined before this call!
  hvd = returnn.tf.horovod.get_ctx(config=config)
  print(prefix, "Local rank/size:", hvd.local_rank(), hvd.local_size())

# Workaround for openblas hanging:
# * https://github.com/tensorflow/tensorflow/issues/13802
# * https://github.com/rwth-i6/returnn/issues/323#issuecomment-725384762
# patch_atfork = not use_horovod
