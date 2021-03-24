
from .lstm import make_lstm
from ..datasets.interface import VocabConfig
from typing import Dict, Any


class Lm:
  vocab: VocabConfig
  opts: Dict[str, Any]
  net_dict: Dict[str, Any]
  model_path: str
