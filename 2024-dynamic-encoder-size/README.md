This folder contains configs and code related to the publication: 

paper [Dynamic Encoder Size Based on Data-Driven Layer-wise Pruning for Speech Recognition]()

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

We use models parts from [i6-models](https://github.com/rwth-i6/i6_models/tree/jing-dynamic-encoder-size) 


### TED-LIUM-v2 Simple-Top-K

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/dynamic_encoder_size/simple_topk_refactored/jointly_train_simple_top_k_layerwise.py)


### TED-LIUM-v2 Iterative-Zero-Out

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/dynamic_encoder_size/iterative_zero_out_refactored/jointly_train_iterative_zero_out_layerwise.py)


### LBS-960 Simple-Top-K

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/lbs_960/pytorch_networks/dynamic_encoder_size/simple_topk/joint_train_three_model_simple_topk_modwise.py)


### LBS-960 Iterative-Zero-Out

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/lbs_960/pytorch_networks/dynamic_encoder_size/zeroout/joint_train_three_model_zeroout_modwise.py)


### LBS-960 Aux-Loss

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/lbs_960/pytorch_networks/baseline/conformer_ctc_d_model_512_num_layers_12_new_frontend_raw_wave_with_aux_loss.py)