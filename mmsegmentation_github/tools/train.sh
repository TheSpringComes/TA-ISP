CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29588 bash tools/dist_train.sh \
    configs/CVPR26_TAISP/taisp_low_mitb0.py 4 \
    --work-dir ./work_dirs/taisp_low_mitb0

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29589 bash tools/dist_train.sh \
    configs/CVPR26_TAISP/taisp_normal_mitb0.py 4 \
    --work-dir ./work_dirs/taisp_normal_mitb0
