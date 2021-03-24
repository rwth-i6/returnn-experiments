

import os


_common_data_path_base = os.environ.get("RETURNN_COMMON_DATA_PATH", "data-common")


def get_common_data_path(filename: str) -> str:
  """
  :param filename: e.g. "librispeech/lm/andre_lstm_bpe1k_lm/net-model/network.020"
  :return: common path for input filename. common across multiple systems.
    The idea is that this can be a symlink to some directory, maybe further including symlinks,
    such that the path can be consistent across multiple systems.

    This would be relative to the current directory, so it should be easy to setup.
    You can overwrite the base path (also to an absolute path) with RETURNN_COMMON_DATA_PATH.

    E.g. return "data-common/librispeech/lm/andre_lstm_bpe1k_lm/net-model/network.020",
    and the user would make a symlink for "data-common".

  This function might be extended by further logic in the future.
  However, any logic here should be fast and lazy,
  and not avoid any real FS interaction (even not stat or so).
  This will be called at config loading time and should not slow down anything
  (consider that even stat can be slow on a NFS or so).
  """
  return os.path.join(_common_data_path_base, filename)
