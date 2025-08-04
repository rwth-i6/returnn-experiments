This folder contains configs and code related to the publication: 

paper [Dynamic Acoustic Model Architecture Optimization in Training for ASR]()

We use [RETURNN](https://github.com/rwth-i6/returnn) for training and our setups are based on [Sisyphus](https://github.com/rwth-i6/sisyphus).

We use models parts from [i6-models](https://github.com/rwth-i6/i6_models/tree/jing-dynamic-encoder-size) 

### DMAO with Conformer CTC

ConformerCTCModel, ConformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/neural_block/dynamic_adaptable_conformer/adapt_based_on_gradient_ranking_finer_granul_double_and_prune.py)

### DMAO with Ebranchformer CTC

EbranchformerCTCModel, EbranchformerCTCConfig and train_step in returnn config is defined in [here](https://github.com/rwth-i6/i6_experiments/blob/main/users/jxu/experiments/ctc/tedlium2/pytorch_networks/neural_block/dynamic_adaptable_e_branchformer/adapt_based_on_gradient_ranking_finer_granul_double_and_prune.py)