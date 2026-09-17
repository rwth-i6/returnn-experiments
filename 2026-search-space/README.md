This folder contains information related to the IEEE SLT 2026 publication:

"A Systematic Comparison of Search Space Complexity and Efficiency for Time-Synchronous End-to-End ASR Models"

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and [RASR](https://github.com/rwth-i6/rasr) for search, and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).
Model parts are taken from [i6_models](https://github.com/rwth-i6/i6_models).

The experiments are designed with Sisyphus in the i6-style experiment format.
The entry points into the experiments are the following config directories in the public i6_experiments repository:

LibriSpeech:
<https://github.com/rwth-i6/i6_experiments/tree/main/users/berger/configs/librispeech/20260917_search_space>

Loquacious:
<https://github.com/rwth-i6/i6_experiments/tree/main/users/berger/configs/loquacious/20260917_search_space>

Both build on the shared training, recognition and data pipelines under
<https://github.com/rwth-i6/i6_experiments/tree/main/users/berger/seq2seq_rasr_2025>

Each config module collects the experiments behind one table or figure of the paper:

- Table I (LibriSpeech WER/RTF under the different search conditions):
  [librispeech/config_01_search_strategies.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/librispeech/20260917_search_space/config_01_search_strategies.py)
- Table II (Loquacious WER/RTF under the different search conditions):
  [loquacious/config_01_search_strategies.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/loquacious/20260917_search_space/config_01_search_strategies.py)
- Tables III and IV (search errors and hypothesis count statistics, BPE CTC and BPE transducer):
  [librispeech/config_02_search_space.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/librispeech/20260917_search_space/config_02_search_space.py)
- Table V (same analysis for the BPE transducer on Loquacious):
  [loquacious/config_02_search_space.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/loquacious/20260917_search_space/config_02_search_space.py)
- Tables VII and VIII (fixed vs. dynamic pruning over the max beam size):
  [librispeech/config_03_dynamic_pruning.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/librispeech/20260917_search_space/config_03_dynamic_pruning.py)
- Tables IX and X (RTF reduction through dynamic pruning, and the tuned decoding parameters):
  [librispeech/config_04_pruning_speedup.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/librispeech/20260917_search_space/config_04_pruning_speedup.py)
  and [loquacious/config_03_pruning_speedup.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/loquacious/20260917_search_space/config_03_pruning_speedup.py)
- Figures 2 and 3 (search RTF vs. WER curves for fixed and dynamic pruning):
  [librispeech/config_05_rtf_vs_wer_curves.py](https://github.com/rwth-i6/i6_experiments/blob/main/users/berger/configs/librispeech/20260917_search_space/config_05_rtf_vs_wer_curves.py)

Table VI and Figure 4 report results on an in-house Spanish telephony corpus which cannot be published, so these experiments are not part of the repository.

For questions, please contact <berger@hltpr.rwth-aachen.de>
