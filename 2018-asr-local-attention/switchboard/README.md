`*.config` files are [RETURNN config files](https://github.com/rwth-i6/returnn),
the training/recog files are in `scores/`.

The following table documents which files correspond to which model:

| Model in paper  | Modelname |
| ------------- | ------------- |
| baseline (global)  | `base.bpe2000`  |
| local (argmax, trained from scratch)  | `local-heuristic.argmax.win{02,05,08,10,12,15,20}.scratch`  |
| local (argmedian, trained from scratch)  | `local-heuristic.median.win{02,05,08,10,12,15,20}.scratch`  |
| softplus position prediction | `local-p.relative.pred256.fixed_sigma2.emb200.softplus.tanh-add.win10.bpe2k.gradnorm5.pretrain.run0` |
| additive combination position prediction | `local-p.relative.sigma-p256.pred256.tanh-add.comb-add.win10.bpe2k.gradnorm5.pretrain.run0` |
| Gaussian position prediction | `local.hou.bpe2k-n.max_step8.win10.adjust_step_sigma.gradnorm5` |
| Scaled gaussian position prediction| `local.tjandra.constrained.win10.max_step10.no_pretrain` |
