This folder contains configs and code related to the publication: 

paper [Dynamic Encoder Size Based on Data-Driven Layer-wise Pruning for Speech Recognition]()

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

We use models parts from [i6-models](https://github.com/rwth-i6/i6_models/tree/jing-dynamic-encoder-size) 


### TED-LIUM-v2 Simple-Top-K

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/dynamic_encoder_size/simple_topk_refactored/jointly_train_simple_top_k_layerwise.py)


### TED-LIUM-v2 Iterative-Zero-Out