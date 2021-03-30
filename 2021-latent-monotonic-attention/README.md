# Configs for `A study of latent monotonic attention variants`

Our experiments were performed with [RETURNN](https://github.com/rwth-i6/returnn),
which is based on [TensorFlow](http://tensorflow.org/).
It should work on Python 2.7 or >=3.6, and using TensorFlow <=1.15.

All the relevant RETURNN configs can be found in the `switchboard` subdirectory.
Please also check `switchboard/README.md` about the organization of these files.

The Switchboard 300h corpus is a corpus which can be acquire from [LDC](https://catalog.ldc.upenn.edu/LDC97S62).
For the feature extraction, we use [RASR](https://www-i6.informatik.rwth-aachen.de/rwth-asr/) (also referred to as Sprint).
We use Switchboard 300h as there is a lot of literature with results on Switchboard, such that it is easy to put the baselines into perspective.

If you don't have access to the Switchboard data, it should be possible to reproduce the results on public corpora such as LibriSpeech or Tedlium.
An full-setup example pipeline using just RETURNN (without RASR) on LibriSpeech can be found [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention). 


## Baselines

The baselines were adopted from [here](https://github.com/rwth-i6/returnn-experiments/blob/master/2019-librispeech-system/attention/base2.conv2l.specaug.curric3.config) (and related configs),
using SpecAugment and initial 2D convolution.


## Monotonic-Latent config structure

The pretraining concept of RETURNN allows for scheduling of custom settings, configurations, network topology, losses, optimizer, learning rate, data, etc, during the training.
This is the `pretrain` option, where we call the function `custom_construction_algo`, which calls `get_net_dict`.
We use that concept during the whole training, to have fine-grained control of the scheduling.
The same function `get_net_dict` also constructs the network for decoding.
Note that one epoch in the config corresponds to a sub-epoch of the training data, as we split the corpora into multiple (6) parts (on-the-fly).

The latent variable `t` is implemented using the [`ChoiceLayer`](https://returnn.readthedocs.io/en/latest/layer_reference/recurrent.html?highlight=ChoiceLayer#returnn.tf.layers.rec.ChoiceLayer).
This allows for the Viterbi search during training (for the best alignment)
and also for the normal beam search during decoding.
[Read more on how beam search with multiple stochastic variables works in RETURNN](https://returnn.readthedocs.io/en/latest/internals/search.html),
and also see our [slides of the Interspeech 2020 tutorial “Efficient and Flexible Implementation of Machine Learning for ASR and MT”](https://www-i6.informatik.rwth-aachen.de/publications/download/1154/Zeyer--2020.pdf) with an introduction of the core concepts,
specifically page 56 about recurrency with multiple stochastic variables.
![image](https://user-images.githubusercontent.com/59132/113063447-9cc20080-91b5-11eb-8746-c013083561eb.png)

In training, we use the [extra-search-net](https://returnn.readthedocs.io/en/latest/api/tf.network.html?highlight=construct_extra_net#returnn.tf.network.TFNetwork.construct_extra_net) concept (`"extra.search:output"` in the config) to enable the search for the best alignment, but fixed ground truth label sequence.

The initial linear alignment is created in the function `t_linear`.
