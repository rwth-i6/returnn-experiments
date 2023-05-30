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
  LOGFILE=rasr.log
fi

export OMP_NUM_THREADS=2
export TF_DEVICE='gpu'

/u/mgunz/src/rs_rasr/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard --config=recognition.config --*.TASK=$TASK --*.LOGFILE=$LOGFILE  $@              