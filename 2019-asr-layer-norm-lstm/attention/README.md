Configs for layer-normalized LSTM variants experiments contains the 
following keywords:
- `gn`: Global Norm
- `gnj`: Global Joined Norm
- `pgn`: Per Gate Norm
- `cn`: Cell Norm


Experiments without pretraining: `*nopre*.config`

Experiments with pretraining: `*pretrain*.config`

Experiments with multiple random seeds: `*rseed*.config`

Experiments with deeper encoders: `*enc*.config`

Experiments with residual connections: `*rnmt*.config`

The attention baseline models are `base.pretrain-start4l-red8.accum2.config` 
and `base.nopre.config`