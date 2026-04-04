#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LOD (BMVC2021) dataset in the same layout as MMDet `LOD_Dataset` / `XMLDataset`."""

import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from loguru import logger

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .lod_classes import LOD_CLASSES


class LODXMLAnnotationTransform(object):
    """VOC-style XML -> numpy boxes [x1,y1,x2,y2,cls] (cls aligned with LOD_CLASSES)."""

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(LOD_CLASSES, range(len(LOD_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            if name not in self.class_to_ind:
                continue
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for pt in pts:
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)
        return res, img_info


class LODXMLDataset(Dataset):
    """
    Paths mirror MMDet config, e.g. R_Net_taisp.py::

        data_root + ann_list_file  -> list of image stems (one per line)
        image: data_root / img_subdir / {stem}.png
        xml:   data_root / ann_subdir / {stem}.xml
    """

    def __init__(
        self,
        data_root,
        ann_list_file,
        img_subdir="JPEGImages",
        ann_subdir="Annotations",
        img_size=(640, 640),
        preproc=None,
        img_ext=".png",
        target_transform=None,
        dataset_name="LOD_XML",
        cache=False,
    ):
        super().__init__(img_size)
        self.data_root = data_root
        self.split_file = os.path.join(data_root, ann_list_file)
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.img_ext = img_ext if img_ext.startswith(".") else "." + img_ext
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform or LODXMLAnnotationTransform()
        self.name = dataset_name
        self._classes = LOD_CLASSES

        with open(self.split_file, "r") as f:
            stems = [ln.strip() for ln in f if ln.strip()]
        self.ids = [(self.data_root, s) for s in stems]

        self.annotations = [self.load_anno_from_ids(i) for i in range(len(self.ids))]
        self.imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _cache_images(self):
        logger.warning(
            "Using cached LOD images in RAM/disk; ensure enough memory and delete cache if input_size changes."
        )
        max_h, max_w = self.img_size[0], self.img_size[1]
        cache_file = self.data_root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            self.imgs = np.memmap(
                cache_file, shape=(len(self.ids), max_h, max_w, 3), dtype=np.uint8, mode="w+"
            )
            from multiprocessing.pool import ThreadPool
            from tqdm import tqdm

            n_threads = min(8, os.cpu_count() or 1)
            loaded = ThreadPool(n_threads).imap(
                lambda x: self.load_resized_img(x), range(len(self.annotations))
            )
            for k, out in tqdm(enumerate(loaded), total=len(self.annotations)):
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
        else:
            self.imgs = np.memmap(
                cache_file, shape=(len(self.ids), max_h, max_w, 3), dtype=np.uint8, mode="r+"
            )

    def load_anno_from_ids(self, index):
        root, stem = self.ids[index]
        xml_path = os.path.join(root, self.ann_subdir, stem + ".xml")
        target = ET.parse(xml_path).getroot()
        res, img_info = self.target_transform(target)
        height, width = img_info
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res = res.copy()
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))
        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_image(self, index):
        """uint8 BGR for `TrainTransform` / `MosaicDetection` (same range as COCO JPEG)."""
        root, stem = self.ids[index]
        img_path = os.path.join(root, self.img_subdir, stem + self.img_ext)
        img = cv2.imread(img_path, -1)
        assert img is not None, img_path
        if img.dtype == np.uint16:
            img = (img.astype(np.float32) * (255.0 / 65535.0)).clip(0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        return cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        )

    def pull_item(self, index):
        # InfiniteSampler yields 0-dim torch.Tensor indices; normalize for indexing and collate.
        index = int(index)
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]
        # Same as COCO: np.ndarray so default_collate stacks [B,1]; avoids Tensor/int mix when
        # MosaicDetection uses random int indices for the last mosaic tile vs tensor idx elsewhere.
        return img, target, img_info, np.array([index])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        # Fifth slot unused by training; VOCEvaluator expects a 5-tuple from the val loader.
        return img, target, img_info, img_id, int(index)

    def _get_voc_results_file_template(self):
        filedir = os.path.join(self.data_root, "results", "VOC", "Main")
        os.makedirs(filedir, exist_ok=True)
        return os.path.join(filedir, "comp4_det_test_{:s}.txt")

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(LOD_CLASSES):
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, _ in enumerate(self.ids):
                    _, stem = self.ids[im_ind]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets is None or (hasattr(dets, "shape") and 0 in dets.shape):
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                stem,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)
        logger.info("map_5095: {:.4f}  map_50: {:.4f}".format(np.mean(mAPs), mAPs[0]))
        return np.mean(mAPs), mAPs[0]

    def _do_python_eval(self, output_dir="output", iou=0.5):
        annopath = os.path.join(self.data_root, self.ann_subdir, "{:s}.xml")
        imagesetfile = self.split_file
        cachedir = os.path.join(self.data_root, "annotations_cache", "VOC", self.name)
        os.makedirs(cachedir, exist_ok=True)
        aps = []
        use_07_metric = False
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(LOD_CLASSES):
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                logger.info("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            logger.info("Mean AP = {:.4f}".format(np.mean(aps)))
        return np.mean(aps)
