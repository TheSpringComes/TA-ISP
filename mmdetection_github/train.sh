#!/usr/bin/env bash
set -euo pipefail

export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  -f lod_yolox_tiny.py \
  -d ${NUM_GPUS:-1} \
  -b ${BATCH_SIZE:-8} \
  -c ${PRETRAIN_CKPT:-./pre-trained/yolox_tiny.pth} \
  -expn ${EXPN_NAME:-lod_yolox_tiny}
