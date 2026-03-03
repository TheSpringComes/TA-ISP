import mmcv
import os.path as osp
import torch

# `PixelData` 是 MMEngine 中用于定义像素级标注或预测的数据结构。
# 请参考下面的MMEngine数据结构教程文件：
# https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html#pixeldata

from mmengine.structures import PixelData

# `SegDataSample` 是在 MMSegmentation 中定义的不同组件之间的数据结构接口，
# 它包括 ground truth、语义分割的预测结果和预测逻辑。
# 详情请参考下面的 `SegDataSample` 教程文件：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

out_file = 'out_file_gt'
save_dir = './work_dirs'
image = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        '/mnt/hpfs1/chenkai/ECCV_RAW_Adapter-main/mmsegmentation_github/data/ADE20K/ADEChallengeData2016/images/validation_raw/ADE_val_00000197.jpg'
    ),
    'color')
sem_seg = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        '/mnt/hpfs1/chenkai/ECCV_RAW_Adapter-main/mmsegmentation_github/data/ADE20K/ADEChallengeData2016/annotations/validation/ADE_val_00000197.png'  # noqa
    ),
    'unchanged')
sem_seg = torch.from_numpy(sem_seg)
gt_sem_seg_data = dict(data=sem_seg)
gt_sem_seg = PixelData(**gt_sem_seg_data)
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir)

# 数据集的元信息通常包括类名的 `classes` 和
# 用于可视化每个前景颜色的 `palette` 。
# 所有类名和调色板都在此文件中定义：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/utils/class_names.py

seg_local_visualizer.dataset_meta = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])

# 当`show=True`时，直接显示结果，
# 当 `show=False`时，结果将保存在本地文件夹中。

seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)