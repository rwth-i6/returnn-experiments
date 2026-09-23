# Text Injection for Speech Recognition

Text injection trains a speech recognition model on unpaired text next to the paired audio.
Here the text is injected at the encoder input: ...

## Code

All code is public in
[i6_experiments](https://github.com/rwth-i6/i6_experiments),
under `users/zeyer/`:

- `experiments/exp2026_05_28_tts_encoder_fzj.py`:
  the [Sisyphus](https://github.com/rwth-i6/sisyphus) recipe (entry point `py()`)
  that builds the whole experiment graph:
  baselines, the injection trainings and their ablations
  (aligner, durations, representation, amount and ratio of text and audio, batch mixing),
  the TTS-based injection for comparison, the recognitions with LM and denoising LM, and the result tables.
  The text-injection model definition and the custom train step
  (audio batches through the front-end, text batches through the table) live here.
- `experiments/exp2026_05_28_tts_encoder.py`:
  shared base (the log-mel front-end, the LibriSpeech base training).
  `exp2026_05_28_tts_encoder_rz.py`, `exp2026_05_28_tts_encoder_rz_torch212.py`:
  the earlier single-GPU variants.
- `experiments/exp2026_05_28_tts_encoder_gauss_hmm.py`:
  the own aligner, a single-Gaussian HMM per phoneme (or HMM state) trained on the ASR features,
  with the table extraction from its alignment.
- `datasets/hf_librispeech_mfa_alignments.py`:
  the MFA alignments of LibriSpeech, the per-phoneme mean log-mel table and the duration statistics.
- `datasets/loquacious.py`, `datasets/loquacious_mfa.py`:
  the Loquacious data (subsets, text-only data, G2P lexicon) and its MFA alignment jobs;
  `experiments/exp2025_07_07_in_grads/jobs/mfa_forced_align.py`, `mfa_native.py`, `proot_qemu.py`:
  running MFA (also on an aarch64 cluster).
- `external_models/glow_tts.py`:
  the frozen GlowTTS wrapper, its lexicon and phoneme vocabulary (used for the phoneme conversion of the text).
- `experiments/exp2024_04_23_baselines/`:
  the model (`aed.py`: Conformer encoder, Transformer decoder, CTC + AED; `configs.py`),
  the optimizer (`optim_ext/muon.py`),
  and the search (`recog_ext/aed_ctc_batched.py`: CTC+AED and CTC+AED+LM;
  `recog_ext/dlm_sum_batched.py`: with the denoising LM).
- `returnn/alternate_batching.py`: the mixing of audio and text batches.
- `utils/table_data.py`: the result tables.

The trainings run through [RETURNN](https://github.com/rwth-i6/returnn) (PyTorch backend),
with packed tensors and CUDA graphs, on 4 GPUs per training.
Jobs are from [i6_core](https://github.com/rwth-i6/i6_core).

## Datasets

- LibriSpeech (960 h paired audio, the LibriSpeech LM corpus as the text, the lexicon for the phonemes).
- Loquacious (25 000 h; subsets as paired audio, the transcripts of the full set as the text).

## Contact

Albert Zeyer, zeyer@ml.rwth-aachen.de
