This folder contains configs and code related to the submitted paper: 

Efficient Supernet Training with Orthogonal Softmax for Scalable ASR Model Compression

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

We use models parts from [i6-models](https://github.com/rwth-i6/i6_models/tree/jing-dynamic-encoder-size) 

### subnet selection with component-wise criterion

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/dynamic_encoder_size/orthogonal_softmax/joint_train_conformer_orthogonal_softmax_component_wise.py)

### subnet selection with layer-wise criterion

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/dynamic_encoder_size/orthogonal_softmax/joint_train_conformer_orthogonal_softmax_layer_wise.py)