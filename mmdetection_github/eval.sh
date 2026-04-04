#!/usr/bin/env bash
# YOLOX 验证入口。默认与 train.sh 使用同一实验文件与 EXPN_NAME，便于对齐输出目录。
#
# 环境变量（均可选）:
#   YOLOX_EXP      实验 py，默认 lod_yolox_tiny.py
#   EXPN_NAME      须与训练时 -expn 一致；默认与 YOLOX_EXP 主文件名一致
#   CKPT           权重路径，默认 ./${EXPN_NAME}/latest_ckpt.pth
#   BATCH_SIZE     默认 8
#   NUM_GPUS       默认 1
#   CUDA_VISIBLE_DEVICES

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

YOLOX_EXP="${YOLOX_EXP:-lod_yolox_tiny.py}"
EXPN_NAME="${EXPN_NAME:-${YOLOX_EXP%.py}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
CKPT="${CKPT:-./${EXPN_NAME}/latest_ckpt.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python eval.py \
  -f "${YOLOX_EXP}" \
  -expn "${EXPN_NAME}" \
  -d "${NUM_GPUS}" \
  -b "${BATCH_SIZE}" \
  -c "${CKPT}" \
  "$@"
