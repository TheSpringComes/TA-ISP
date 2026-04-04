#!/usr/bin/env bash
# YOLOX 训练入口。默认实验：LOD MMDet VOC-XML（lod_yolox_tiny.py）。
#
# 环境变量（均可选）:
#   YOLOX_EXP      实验 py，默认 lod_yolox_tiny.py（COCO 版 LOD 用 lod_coco_format_yolox_tiny.py）
#   EXPN_NAME      输出子目录名，默认与 YOLOX_EXP 主文件名一致
#   BATCH_SIZE     默认 16
#   NUM_GPUS       默认 1
#   PRETRAIN_CKPT  默认 ./pre-trained/yolox_tiny.pth
#   CUDA_VISIBLE_DEVICES

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

YOLOX_EXP="${YOLOX_EXP:-lod_yolox_tiny.py}"
EXPN_NAME="${EXPN_NAME:-${YOLOX_EXP%.py}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-./pre-trained/yolox_tiny.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  -f "${YOLOX_EXP}" \
  -d "${NUM_GPUS}" \
  -b "${BATCH_SIZE}" \
  -c "${PRETRAIN_CKPT}" \
  -expn "${EXPN_NAME}" \
  "$@"
