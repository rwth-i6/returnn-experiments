This folder contains information related to the publication :

"Analyzing the Importance of Blank for CTC-Based Knowledge Distillation"

The full experimental setup can be found in public i6-experiment repository.

The initial hooks for the corresponding experiments and model setup are as follows:

-- TEDLIUM
https://github.com/rwth-i6/i6_experiments/blob/main/users/hilmes/experiments/tedlium2/standalone/experiments/ctc_phon/finetune_hubert.py

-- LibriSpeech
https://github.com/rwth-i6/i6_experiments/blob/main/users/hilmes/experiments/librispeech/ctc_rnnt_standalone_2024/experiments/ctc_phon/distill_baselines.py

Please note that most of the code was not specifically cleaned up and may contain unrelated experiments. For questions on how to run experiments and setup Sisyphus, please open an issue here or write to [hilmes|rossenbach]@cs.rwth-aachen.de
