# MNIST classification task in RETURNN

This setup should serve as a simplest application of RETURNN.
MNIST data provided as a RETURNN dataset packed in HDF file.
Train and test sets can be downloaded here: [mnist.train.h5](http://ikulikov.name/mnist.train.h5), [mnist.test.h5](http://ikulikov.name/mnist.test.h5)

## Download data

`./mnist-download.sh`

## Prepare RETURNN

`git clone https://github.com/rwth-i6/returnn.git`

## Check the config file for training

`less config/ff_3l_sgd.config`

## Start training

`./returnn/rnn.py config/ff_3l_sgd.config`
