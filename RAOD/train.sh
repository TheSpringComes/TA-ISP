export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

python main.py \
    -f cfg/cfg_lod_tiny.py \
    -c pre-trained/yolox_tiny.pth \
    -expn experiments/LOD_taisp_tiny_day