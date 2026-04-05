#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright 2023 Huawei Technologies Co., Ltd
#
# =============================================================================
# LOD 训练配置说明（请先读这一段）
# =============================================================================
#
# 1) 和 ROD（cfg/cfg_tiny.py）有什么区别？
#    - ROD：COCO JSON + gzip numpy 图，用 COCORawDataset，验证走 COCOEvaluator。
#    - LOD：VOC 式 XML + 普通图像文件，用 LODDetection，验证走 VOCEvaluator（mAP 与 VOC 脚本一致）。
#    - 类别数：LOD 为 8（lod_classes.py），与 mmdet 的 LOD_Dataset 一致；ROD 为 5。
#
# 2) 为什么改了 yolox/data/datasets/lod.py？
#    原版 LODDetection 写死：ImageSets/Main/{name}.txt、JPEGImages、Annotations。
#    而 mmdet 的 R_Net_taisp.py 使用的是（与当前仓库内该文件保持同步）：
#      data_root = /home/jing/datasets/LOD_BMVC2021/LOD_BMVC2021/
#      ann_file  = trainval/train.txt 或 trainval/val.txt
#      img_subdir = RAW_dark
#      ann_subdir = RAW-dark-Annotations
#    因此在 LODDetection 里增加了可选参数：image_list_file、img_subdir、ann_subdir、img_ext。
#    不设 image_list_file 时，行为与旧版 VOC 布局完全兼容。
#
# 3) 数据应长什么样？
#    以 BMVC2021 为例（与 R_Net_taisp 一致）：
#      LOD_BMVC2021/
#        trainval/train.txt          # 每行一个样本 id（无扩展名）
#        trainval/val.txt
#        RAW_dark/{id}.png           # 实际扩展名若不是 .png，改 lod_img_ext
#        RAW-dark-Annotations/{id}.xml
#
# 4) data_dir
#    请与 R_Net_taisp.py 里的 data_root 一致。当前为绝对路径；若你本机路径不同，只改 self.data_dir。
#
# 5) 训练命令示例（YOLOX-Tiny，width=0.375）
#    python main.py -f cfg/cfg_lod_tiny.py -b 64 -c pre-trained/yolox_tiny.pth
#
# 5b) 仅训练 TAISP + 分类支路（冻结 YOLOX backbone 与检测头的 stem/reg/obj）
#    self.freeze_pretrained_yolox_except_isp_and_cls = True
#    在 resume_train 加载 -c 预训练权重之后生效；False 为全参数训练。
#
# 6) 与 mmdet 其它项
#    R_Net_taisp 里 max_epochs=10、train 使用 RepeatDataset×8 等仅作用于 mmdet 调度；
#    本 cfg 仍沿用 YOLOX 默认 max_epoch=300；若要对齐短训可把 max_epoch 改为 10。
#
# =============================================================================

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as YoloXBaseExp

_RAOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Exp(YoloXBaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- model config (YOLOX-Tiny) ---------------- #
        self.num_classes = 8
        self.depth = 0.33
        self.width = 0.375
        self.act = 'silu'
        self.gamma_range = [2.0, 6.0]

        # ---------------- LOD 数据路径（与 R_Net_taisp.py data_root 同步）---------------- #
        self.data_dir = '/home/jing/datasets/LOD_BMVC2021/LOD_BMVC2021'
        self.lod_image_list_train = 'trainval/train.txt'
        self.lod_image_list_val = 'trainval/val.txt'
        self.lod_image_list_test = 'trainval/val.txt'
        self.lod_img_subdir = 'RAW_dark'
        self.lod_ann_subdir = 'RAW-dark-Annotations'
        self.lod_img_ext = '.png'
        self.lod_dataset_name = 'LOD_BMVC2021_RAW_dark'

        # ---------------- dataloader 其它 ---------------- #
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.multiscale_range = 5

        # ---------------- transform config ---------------- #
        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        # ---------------- training config ---------------- #
        self.warmup_epochs = 1
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.freeze_pretrained_yolox_except_isp_and_cls = False

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 5
        self.output_dir = _RAOD_ROOT
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split('.')[0]

        # ---------------- testing config ---------------- #
        self.test_size = (640, 640)
        self.test_conf = 0.001
        self.nmsthre = 0.65
        self.seed = 0

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]

            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=False)
            head = YOLOXHead(self.num_classes, self.width, strides=[16, 32, 64], in_channels=in_channels, act=self.act, depthwise=False)
            self.model = YOLOX(backbone, head, nf=16, gamma_range=self.gamma_range)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 64) - self.multiscale_range
                max_size = int(self.input_size[0] / 64) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def get_data_loader(self, batch_size, is_distributed, no_aug=True, cache_img=False):
        from yolox.data import LODDetection, TrainTransformRaw, YoloBatchSampler
        from yolox.data import DataLoader, InfiniteSampler, worker_init_reset_seed
        from yolox.utils import wait_for_the_master, get_local_rank

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = LODDetection(
                data_dir=self.data_dir,
                image_list_file=self.lod_image_list_train,
                img_subdir=self.lod_img_subdir,
                ann_subdir=self.lod_ann_subdir,
                img_ext=self.lod_img_ext,
                img_size=self.input_size,
                preproc=TrainTransformRaw(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                dataset_name=self.lod_dataset_name,
                cache=cache_img,
            )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import LODDetection, ValTransformRaw

        list_file = self.lod_image_list_test if testdev else self.lod_image_list_val
        valdataset = LODDetection(
            data_dir=self.data_dir,
            image_list_file=list_file,
            img_subdir=self.lod_img_subdir,
            ann_subdir=self.lod_ann_subdir,
            img_ext=self.lod_img_ext,
            img_size=self.test_size,
            preproc=ValTransformRaw(legacy=legacy),
            dataset_name=self.lod_dataset_name,
            cache=False,
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )


if __name__ == '__main__':
    e = Exp()
    d = e.get_data_loader(batch_size=4, is_distributed=False, no_aug=False, cache_img=False)
    for (iteration, a) in enumerate(d):
        print(len(a))
        for item in a[:2]:
            print(type(item), torch.max(item), item.shape)
