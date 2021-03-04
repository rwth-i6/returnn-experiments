
from ...asr.gt import make_returnn_audio_features_func


def make_gt_features_opts(dim=50):
  """
  Use this for `audio` options in e.g. `OggZipDataset`.
  """
  return {
    "peak_normalization": False,
    "features": make_returnn_audio_features_func(),
    "num_feature_filters": dim}
