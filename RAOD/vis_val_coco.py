#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os

import cv2
import torch
from tqdm import tqdm

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.visualize import vis


def make_parser():
    parser = argparse.ArgumentParser("YOLOX COCO val visualization")
    parser.add_argument(
        "-f", "--exp_file", required=True, type=str, help="experiment description file"
    )
    parser.add_argument(
        "-c", "--ckpt", required=True, type=str, help="checkpoint file for visualization"
    )
    parser.add_argument(
        "--save_dir",
        default="vis_val_coco",
        type=str,
        help="directory to save rendered validation images",
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="inference device")
    parser.add_argument(
        "--vis_conf",
        default=0.25,
        type=float,
        help="only draw boxes above this score threshold",
    )
    parser.add_argument(
        "--test_conf",
        default=None,
        type=float,
        help="override experiment test confidence before NMS",
    )
    parser.add_argument(
        "--nms",
        default=None,
        type=float,
        help="override experiment NMS threshold",
    )
    parser.add_argument(
        "--max_images",
        default=None,
        type=int,
        help="maximum number of validation images to visualize",
    )
    return parser


def main():
    args = make_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    exp = get_exp(args.exp_file, None)
    if args.test_conf is not None:
        exp.test_conf = args.test_conf
    if args.nms is not None:
        exp.nmsthre = args.nms

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(args.device)

    val_loader = exp.get_eval_loader(
        batch_size=1, is_distributed=False, testdev=False, legacy=False
    )
    dataset = val_loader.dataset
    class_names = dataset._classes

    num_images = len(dataset) if args.max_images is None else min(len(dataset), args.max_images)

    with torch.no_grad():
        for idx in tqdm(range(num_images), desc="Visualizing val images"):
            img_tensor, _, img_info, _, file_name = dataset[idx]
            inp = torch.from_numpy(img_tensor).unsqueeze(0).float().to(args.device)

            outputs = model(inp)
            outputs = postprocess(
                outputs,
                exp.num_classes,
                exp.test_conf,
                exp.nmsthre,
            )

            rendered = dataset.load_image(idx).copy()

            if outputs[0] is not None:
                output = outputs[0].cpu()
                bboxes = output[:, 0:4].clone()

                img_h, img_w = img_info
                scale = min(
                    exp.test_size[0] / float(img_h),
                    exp.test_size[1] / float(img_w),
                )
                bboxes /= scale

                scores = (output[:, 4] * output[:, 5]).numpy()
                cls_ids = output[:, 6].numpy()

                rendered = vis(
                    rendered,
                    bboxes.numpy(),
                    scores,
                    cls_ids,
                    conf=args.vis_conf,
                    class_names=class_names,
                )

            cv2.imwrite(os.path.join(args.save_dir, file_name), rendered)


if __name__ == "__main__":
    main()
