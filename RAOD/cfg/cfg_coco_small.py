#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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
        # ---------------- model config (YOLOX-S) ---------------- #
        # 标准 COCO 为 80 类；若你是自定义 COCO，请按实际类别数修改
        self.num_classes = 80
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"
        self.gamma_range = [2.0, 6.0]

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.multiscale_range = 5

        self.data_dir = "/home/jing/datasets/COCO"
        self.train_ims = "train2017_SynRAW"
        self.val_ims = "val2017_SynRAW"

        # COCODataset 内部固定拼接: os.path.join(data_dir, "annotations", json_file)
        # 用 ../ 回到 data_dir 后再进入你的标注目录，避免改动底层代码
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = self.val_ann
        self.test_ims = self.val_ims

        # ---------------- transform config ---------------- #
        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0
        self.mixup_prob = 0.0
        self.mixup_scale = (0.5, 1.5)

        # ---------------- training config ---------------- #
        self.warmup_epochs = 1
        self.max_epoch = 30
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
        self.output_dir = _RAOD_ROOT
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

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
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        return (tensor[0].item(), tensor[1].item())

    def get_data_loader(self, batch_size, is_distributed, no_aug=True, cache_img=False):
        from yolox.data import COCODataset, TrainTransform, YoloBatchSampler
        from yolox.data import DataLoader, InfiniteSampler, worker_init_reset_seed
        from yolox.utils import wait_for_the_master, get_local_rank

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.train_ims,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
            )

        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_ims if not testdev else self.test_ims,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}
        dataloader_kwargs["batch_size"] = batch_size
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
