Configs and recipes for our global attention-based encoder-decoder models.

Paper: [The Conformer Encoder May Reverse the Time Dimension](https://arxiv.org/abs/2410.00680)

We use [RETURNN](https://github.com/rwth-i6/returnn) based on PyTorch for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

### Sisyphus Setup

To prepare a sisyphus setup, you can run `prepare_sis_dir.sh <setup-dirname>`.
After that, you need to create a `__init__.py` file inside
the `config` folder and import the sis config there to run it.
Here is an example to run chunked AED experiments for librispeech:
```python
from i6_experiments.users.schmitt.experiments.exp2024_08_27_flipped_conformer import flipped_conformer_exps


def main():
  flipped_conformer_exps.py()
```
Then, to run the experiments, you just call `./sis m` inside the setup dir. This
will call the `main` function defined above.
