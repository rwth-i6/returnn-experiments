
# Needed for Theano.
# Prefer CUDA 7.0 if available. Needed for new GPUs.
# If Sprint also uses CUDA in the same proc, it needs to use the same CUDA version!
# See https://groups.google.com/forum/#!topic/theano-users/Pu4YKlZKwm4
source /u/zeyer/py-envs/py2-theano-new/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/atlas-base/

export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH  # also needed for sprint
export LD_LIBRARY_PATH=/usr/local/cuda-6.5.19/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-6.5.19/bin:$PATH

#export PATH=/usr/local/cuda-7.0/bin:$PATH

#export THEANO_FLAGS="device=gpu,force_device=True"
export THEANO_FLAGS="base_compiledir=/var/tmp/$LOGNAME/theano/,compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s--gpu"
export THEANO_FLAGS="$THEANO_FLAGS,lib.cnmem=1"

# For profiling:
# Debugging http://stackoverflow.com/questions/28221191/theano-cuda-error-an-illegal-memory-access-was-encountered
#export THEANO_FLAGS="$THEANO_FLAGS,profile=True,profiling.time_thunks=1"
#export CUDA_LAUNCH_BLOCKING=1 # Makes all GPU ops synchronized. They are async by default.

# Newer GCC. Old GCC is buggy and breaks Theano on new CPUs.
# https://groups.google.com/forum/#!msg/theano-users/iNNxMqDIJjY/KSbvXlGOV9AJ
# https://github.com/Theano/Theano/issues/1980
export PATH=/work/speech/tools/gcc-4.8.0/x64/gcc-4.8.0/bin:$PATH
export LD_LIBRARY_PATH=/work/speech/tools/gcc-4.8.0/x64/gcc-4.8.0/lib64:/work/speech/tools/gcc-4.8.0/x64/cloog-0.18.0/lib:/work/speech/tools/gcc-4.8.0/x64/gmp-5.1.2/lib:/work/speech/tools/gcc-4.8.0/x64/isl-0.11.1/lib:/work/speech/tools/gcc-4.8.0/x64/mpc-1.0.1/lib:/work/speech/tools/gcc-4.8.0/x64/mpfr-3.1.2/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/work/speech/tools/gcc-4.8.0/x64/gcc-4.8.0/lib/gcc/x86_64-unknown-linux-gnu/4.8.0:$LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
