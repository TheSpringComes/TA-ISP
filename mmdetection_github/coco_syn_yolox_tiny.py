#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO 合成「RAW」管线（与 `COCO_Syn_preprocess.py` + MMDet 一致）在 YOLOX 上的配置。

图像：与 `configs/COCO/COCO_R_Net_taisp.py` 相同，来自 `train2017_SynRAW/`、`val2017_SynRAW/` 等目录下的
`.jpg`（或 `.png`），经 `cv2.imread` 后 **/255.0**，与 MMDet `DetDataPreprocessor(mean=0, std=255)` 及
本仓库 `YOLOX`/`TAISP` 期望的 **约 [0,1]** 输入一致。

数据增强：使用 `TrainTransformRaw` + `MosaicDetectionRaw`（与 RAOD `COCORawDataset` 相同数值域），
**不要** 对 Syn 数据使用标准 `TrainTransform`（0~255 float，不适合 TAISP）。
"""

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as YoloXBaseExp


class Exp(YoloXBaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.depth = 0.33
        self.width = 0.375
        self.act = "silu"
        self.gamma_range = [4.0, 6.0]

        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.multiscale_range = 5

        # 与 configs/COCO/COCO_R_Net_taisp.py 对齐（按本机修改 data_root）
        self.data_dir = "/home/jing/datasets/COCO/"
        self.train_ann = "annotations_trainval2017/annotations/instances_train2017.json"
        self.val_ann = "annotations_trainval2017/annotations/instances_val2017.json"
        self.train_ims = "train2017_SynRAW"
        self.val_ims = "val2017_SynRAW"
        self.ann_folder = ""
        self.image_backend = "jpeg"

        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        self.warmup_epochs = 1
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 5
        self.output_dir = os.path.dirname(__file__)
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

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
            from models import YOLOPAFPN, YOLOX, YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=False
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                strides=[16, 32, 64],
                in_channels=in_channels,
                act=self.act,
                depthwise=False,
            )
            self.model = YOLOX(backbone, head, nf=16, gamma_range=self.gamma_range)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, "random_size"):
                min_size = int(self.input_size[0] / 64) - self.multiscale_range
                max_size = int(self.input_size[0] / 64) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0], tensor[1] = size[0], size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)
        return (tensor[0].item(), tensor[1].item())

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCORawDataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetectionRaw,
            TrainTransformRaw,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import get_local_rank, wait_for_the_master

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            base = COCORawDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_ims,
                img_size=self.input_size,
                preproc=TrainTransformRaw(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
                ann_folder=self.ann_folder,
                image_backend=self.image_backend,
            )

        dataset = MosaicDetectionRaw(
            base,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformRaw(
                max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=(0.5, 1.5),
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=0.0,
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug
        )
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCORawDataset, ValTransformRaw

        valdataset = COCORawDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.val_ims,
            img_size=self.test_size,
            preproc=ValTransformRaw(legacy=legacy),
            ann_folder=self.ann_folder,
            image_backend=self.image_backend,
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(
            valdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )
