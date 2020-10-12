#!/usr/bin/env bash

echo "activating CUDA"

# If you are not executing Sisyphus with a cuda env, you can add it here
#source /path/to/your/env/bin/activate
export PS1=""

# Set CUDA and CUDNN path according to your installation
export CUDA_HOME=/path/to/cuda
export CUDNN=/path/to/cudnn

# change if necessary
export NVIDIA_DRIVER=/usr/lib/nvidia-418

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDNN/lib64:$NVIDIA_DRIVER"

# Uncomment and set ID to limit execution to a specific GPU
# export CUDA_VISIBLE_DEVICES=0

echo $PYTHONPATH
echo "$@"
python $*

