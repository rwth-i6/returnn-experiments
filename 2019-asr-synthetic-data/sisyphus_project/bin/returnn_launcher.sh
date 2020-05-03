#!/usr/bin/env bash

echo "activating CUDA"
# If you are not executing Sisyphus with a cuda env, you can add it here
# source /path/to/your/cuda/env/bin/activate
export PS1="" 
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cudnn-10.0-v7.4/lib64:/usr/lib/nvidia-418" 
export CUDA_HOME=/usr/local/cuda-10.0/ 
export CUDNN=/usr/local/cudnn-10.0-v7.4/

# Uncomment and set ID to limit execution to a specific GPU
# export CUDA_VISIBLE_DEVICES=0

echo $PYTHONPATH
echo "$@"
python $*

