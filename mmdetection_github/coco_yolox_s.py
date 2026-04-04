#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard MS COCO detection with YOLOX-S-style depth/width (JPEG + COCO JSON)."""

import os

from yolox.exp import Exp as YoloXBaseExp


class Exp(YoloXBaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        self.data_dir = os.path.join(os.path.expanduser("~"), "datasets", "coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "image_info_test-dev2017.json"
        self.train_ims = "train2017"
        self.val_ims = "val2017"
        self.test_ims = "test2017"
        self.ann_folder = "annotations"

        self.output_dir = os.path.dirname(__file__)
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.seed = 0

        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.enable_mixup = True
