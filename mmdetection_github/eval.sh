#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python eval.py \
  -f lod_yolox_tiny.py \
  -d ${NUM_GPUS:-1} \
  -b ${BATCH_SIZE:-8} \
  -c ${CKPT:-./lod_yolox_tiny/latest_ckpt.pth}
