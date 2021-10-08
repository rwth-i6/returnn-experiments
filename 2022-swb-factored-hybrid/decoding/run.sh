#!/usr/bin/env bash
set -ueo pipefail

if [[ $# -gt 0 ]]; then
  TASK=$1;
  shift;
else
  echo "No TASK-id given";
  exit 1;
fi

if [ $# -gt 0 ]; then
  LOGFILE=$1;
  shift;
else
  LOGFILE=sprint.log
fi

:${THEANO_FLAGS:=}
export THEANO_FLAGS="$THEANO_FLAGS,device=gpu,force_device=True"
export TF_DEVICE="gpu"

/u/raissi/dev/rasr-dense/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard --config=recognition.config --*.TASK=$TASK --*.LOGFILE=$LOGFILE $@              