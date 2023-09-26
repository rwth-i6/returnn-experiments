Configs and recipes for the chunked attention-based encoder-decoder model.

Paper: [Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition](https://arxiv.org/abs/2309.08436)

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

### Global AED baselines

Generated configs via sisyphus can be found in `{librispeech,ted2}/global_aed_baseline.py`.

Sisyphus configs can be also found here:
- LibriSpeech: https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/experiments/conformer_att_2022/librispeech_960/configs/baseline_960h_v2.py
- TED-LIUM-v2: https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/experiments/conformer_att_2023/tedlium2/configs/ted2_att_baseline.py

### Chunked AED Experiments

Here are the sisyphus configs:
- LibriSpeech: https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/experiments/chunkwise_att_2023/librispeech_960/configs/chunked_aed.py
- TED-LIUM-v2:

We use this [script](https://github.com/rwth-i6/i6_experiments/blob/main/users/zeyer/experiments/exp2023_02_16_chunked_attention/scripts/latency.py)
for word emit latency measure. There is also a sisyphus job for that [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/experiments/chunkwise_att_2023/latency.py#L5).

### Sisyphus Setup

To prepare a sisyphus setup, you can run `prepare_sis_dir.sh <setup-dirname>`.
After that, you need to create a `__init__.py` file inside
the `config` folder and import the sis config there to run it.
Here is an example to run chunked AED experiments for librispeech:
```python
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.configs.chunked_aed import run_all_exps

def main():
    run_all_exps()
```