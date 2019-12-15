#!/usr/bin/env bash

echo "activating CUDA 10"
source /work/iwslt/smt/rossenbach/returnn_env_cuda10/bin/activate
export PS1="" 
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cudnn-10.0-v7.4/lib64:/usr/lib/nvidia-418" 
export CUDA_HOME=/usr/local/cuda-10.0/ 
export CUDNN=/usr/local/cudnn-10.0-v7.4/ 

echo $PYTHONPATH
echo "$@"
python $*

