# Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs

Albert Zeyer, Ralf Schlüter, Hermann Ney. IEEE SLT 2026.

Paper: https://arxiv.org/abs/2607.06831

The alignment takes the gradient of each teacher-forced token log probability
w.r.t. the input (or an intermediate encoder layer),
reduces it to a per-frame saliency,
and decodes the token-by-frame matrix into word boundaries with a dynamic-programming pass.
It needs no training and no model modification.

## Code

All code is public in
[i6_experiments](https://github.com/rwth-i6/i6_experiments),
under `users/zeyer/experiments/`:

- `exp2026_05_23_grad_align.py`:
  the [Sisyphus](https://github.com/rwth-i6/sisyphus) recipe (entry point `py()`)
  that builds the whole experiment graph of the paper:
  all models, datasets, alignments, baselines, ablations, tables and figures.
- `exp2026_05_23_grad_align_speedcmp_p212.py`:
  companion recipe for the cost table (torch 2.12, compiled CTC prefix-score lattice).
- `exp2025_07_07_in_grads/jobs/`: the Sisyphus jobs,
  - `extract_per_token_grads.py`: per-token gradient extraction
    (`ExtractInGradsPerTokenJob`; batched backward, SmoothGrad / IG / EG variants),
  - `word_align_from_per_token_grads.py`: the DP decoder into word boundaries
    (blank schemes, energy weighting, topologies) and the metrics (WBE, accuracy at collar),
  - `models/`: one adapter per model
    (`whisper.py`, `wav2vec2_ctc.py`, `wav2vec2_phoneme_ctc.py`, `parakeet_ctc.py`,
    `parakeet_rnnt.py` (RNN-T and TDT), `emformer_rnnt.py`, `fastconformer_streaming.py`,
    `owsm_ctc.py`, `owls.py`, `voxtral.py`, `phi4mm.py`, `canary_qwen.py`),
    with the CTC prefix scores in `ctc_partial.py` and the transducer prefix scores in the transducer adapters,
  - baselines: `forced_align_baseline.py` (MMS-FA), `mfa_forced_align.py` (MFA via Apptainer),
    `native_transducer_align.py`, `parakeet_ctc_forced_align.py`, `owsm_ctc_forced_align.py`,
    `phoneme_forced_align_baseline.py`,
    `whisper_crossattn_align.py` and `extract_self_attn.py` (attention alignments, head selection),
    `crisper_whisper_align.py` (official CrisperWhisper pipeline),
  - `hyp_align.py`, `recog_from_model.py`: hypothesis-mode alignment,
  - `buckeye_fine_dataset.py`: the Buckeye segmentation,
  - `param_noise.py`, `perturb.py`, `wer_noise_sweep.py`: robustness experiments,
  - `align_cost_benchmark.py`, `align_cost_4way_benchmark.py`: the cost table,
  - `grad_align_tables.py`, `grad_align_plots.py`, `figure_builders.py`: the paper tables and figures.
- `exp2025_05_05_align.py`: the `Aligner` class (the DP over the saliency matrix).

The models run through [RETURNN](https://github.com/rwth-i6/returnn) (PyTorch backend)
or directly through their own frameworks (HuggingFace transformers, NeMo, ESPnet, torchaudio).

## Datasets

- TIMIT (test, and dev for the attention-head selection), gold word boundaries.
- Buckeye, 5 h subset stratified by speaker,
  split at inter-word silences of at least 1 s and at the largest internal gap above 18 s,
  gold word boundaries. The segmentation is in `buckeye_fine_dataset.py`.

## Contact

Albert Zeyer, zeyer@ml.rwth-aachen.de
