#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
LOD BMVC2021 + YOLOX（**默认：MMDet VOC-XML**，与 ``configs/LOD/R_Net_taisp.py`` 一致）。

文件分工
--------
* **本文件** — 官方 LOD 布局：`train.txt` / `val.txt` + XML + `RAW_dark` 等（``train.sh`` 默认 ``-f lod_yolox_tiny.py``）。
* **``lod_coco_format_yolox_tiny.py``** — 仅当你把 LOD 转成 COCO JSON（``instances_*.json``）时使用。

目录结构（XML 默认）
--------------------
``data_dir``/``trainval/train.txt``、``val.txt`` — 每行一个图像 stem（无后缀）  
``data_dir``/``{img_subdir}``/``{stem}{img_ext}`` — 图像  
``data_dir``/``{ann_subdir}``/``{stem}.xml`` — 标注  

默认 ``img_subdir=RAW_dark``，``ann_subdir=RAW-dark-Annotations``，``img_ext=.png``；按本机修改 ``self.data_dir`` 等字段。
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
        self.num_classes = 8
        self.depth = 0.33
        self.width = 0.375
        self.act = "silu"
        self.gamma_range = [4.0, 6.0]

        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.multiscale_range = 5

        # 与 configs/LOD/R_Net_taisp.py 一致（按本机修改）
        self.data_dir = "/home/jing/datasets/LOD_BMVC2021/LOD_BMVC2021/"
        self.train_list = "trainval/train.txt"
        self.val_list = "trainval/val.txt"
        self.test_list = "trainval/val.txt"
        self.img_subdir = "RAW_dark"
        self.ann_subdir = "RAW-dark-Annotations"
        self.img_ext = ".png"

        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        self.warmup_epochs = 1
        self.max_epoch = 100
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 50
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
            DataLoader,
            InfiniteSampler,
            LODXMLDataset,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import get_local_rank, wait_for_the_master

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            base = LODXMLDataset(
                data_root=self.data_dir,
                ann_list_file=self.train_list,
                img_subdir=self.img_subdir,
                ann_subdir=self.ann_subdir,
                img_size=self.input_size,
                preproc=None,
                img_ext=self.img_ext,
                cache=cache_img,
            )

        dataset = MosaicDetection(
            base,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
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
        from yolox.data import LODXMLDataset, ValTransform

        list_file = self.val_list if not testdev else self.test_list
        valdataset = LODXMLDataset(
            data_root=self.data_dir,
            ann_list_file=list_file,
            img_subdir=self.img_subdir,
            ann_subdir=self.ann_subdir,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            img_ext=self.img_ext,
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
