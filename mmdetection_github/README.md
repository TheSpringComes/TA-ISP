# TA-ISP Object Detection (LOD dataset, PASCAL RAW dataset)

### 📖 1: Dataset Download

**LOD Dataset** (low-light RAW detection dataset):

Download LOD dataset  from [Google Drive](https://drive.google.com/file/d/1Jkm4mvynWxc7lXSH3H9sLI0wJ6p6ftvZ/view?usp=sharing) or [百度网盘 (passwd: kf43)](https://pan.baidu.com/s/1FA9lw1WXk2dJ0jtlLeho5w), or find the LOD dataset [original provide link](https://github.com/ying-fu/LODDataset).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- LOD_BMVC21
         -- RAW_dark (RAW data, "demosacing" in our paper)
         -- RGB_dark (default ISP RGB data)
         -- RAW_dark_InverseISP (InvISP processed RAW data, [CVPR 2021])
         -- RAW_dark_ECCV16_Micheal (ECCV16 ISP processed RAW data, [ECCV 2016])
         -- RAW-dark-Annotations (detection label)      
         -- trainval
```

**PASCAL RAW Dataset** (RAW detection dataset):

Download LOD dataset from [Google Drive](https://drive.google.com/file/d/1686W89ALVvtfUvK8NMvqWaUCTLBqhW-p/view?usp=sharing) or [百度网盘 (passwd: kjv9)](https://pan.baidu.com/s/1O76R8ZFZdLw88N0b3hT2Tw).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- PASCAL_RAW
         -- annotations
         -- original (original RAW, demosaic RAW normal-light & over-exposure & low-light)
         -- compare_ISP (ISP methods, InvISP, ECCV16-ISP)
         -- trainval
```

**Note:** the original RAW file in PASCAL RAW are too big (>100GB), you could download them from [PASCAL RAW webiste](https://purl.stanford.edu/hq050zr7488), the code translate original RAW data to demosaic RAW data (normal-light, over-exposure, low-light) could find in here: [PASCAL_RAW_pre_process.py](PASCAL_RAW_pre_process.py).


### 📖 2: Enviroment Setup

Our code based on [mmdetection 3.3.0](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file) version, you can following their [instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html) to build environment. Or following our steps below:

(1). Create a conda environment and activate it:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

(2). Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. ours (torch1.12.1+cu113):
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

(3). Mmdetection setup:
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Develop mmdet and install rawpy:
```
pip install -v -e .
pip install rawpy
```

### 📖 3: Model Evaluation

Check the whole pretrain weights at [release](https://github.com/CVL-UESTC/TA-ISP/releases).

#### 3.1: **LOD Dataset** config & pretrain weights (ckpt).

RetinaNet - ResNet50 backbone: 

| **TA-ISP** | 63.9 | [config](configs/LOD/R_Net_taisp.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/lod_RNet_res50_taisp.pth)


Evaluation of TA-ISP, only need single GPU (TA-ISP, RetinaNet for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/LOD/R_Net_taisp.py https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/lod_RNet_res50_taisp.pth
```

#### 3.2: **PASCAL RAW Dataset** config & pretrain weights (ckpt):

RetinaNet - ResNet18 backbone: 

|  **TA-ISP** | 89.9 | [config](configs/PASCALRAW_Res18/Normal_Light_taisp_res18.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res18_taisp.pth)

RetinaNet - ResNet50 backbone: 

| **TA-ISP** | 90.2 | [config](configs/PASCALRAW_Res50/Normal_Light_raw_adapter_res50.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res50_taisp.pth)

Evaluation of TA-ISP, only need single GPU (TA-ISP, ResNet18), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/PASCALRAW_Res18/Normal_Light_taisp_res18.py https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res18_taisp.pth
```

### 📖 4: Model Training (Optional)

MMDet 训练请使用官方入口，例如：

```bash
bash tools/dist_train.sh <config> <work_dir>
```

根目录下的 **`train.sh`** 属于本仓库自带的 **YOLOX** 流程（默认 LOD XML：`lod_yolox_tiny.py`），与 `tools/train.sh` 不是同一个脚本。


### 📖 Acknowledgement:

We thanks mmdetection & LOD & PASCAL RAW for their excellent code base & dataset.


## YOLOX 实验（与 RAOD 同款工程结构）

工程入口：`main.py`、`train.sh`、`eval.sh`、`models/`、`yolox/`、`scripts/`。

### LOD（默认 XML，与 MMDet 一致）

| 文件 | 用途 |
|------|------|
| **`lod_yolox_tiny.py`** | **默认**：VOC-XML + `trainval/train.txt` / `val.txt`，同 `configs/LOD/R_Net_taisp.py`；验证 `VOCEvaluator`。`train.sh` 已写 `-f lod_yolox_tiny.py`。 |
| **`lod_coco_format_yolox_tiny.py`** | 可选：LOD 已转成 COCO JSON（`annotations/instances_*.json` + `train/`）时用。 |

**XML 默认目录示例**（在 `lod_yolox_tiny.py` 里改 `data_dir`）：

```
<LOD_ROOT>/
  trainval/train.txt
  trainval/val.txt
  RAW_dark/*.png
  RAW-dark-Annotations/*.xml
```

**若使用 COCO 版 LOD**（与上面二选一）：

```
./data/LOD_BMVC2021_COCO/
  annotations/instances_train.json
  annotations/instances_val.json
  train/
  val/
```

### 其它 exp

- `coco_yolox_s.py`：MS COCO 2017 路径示例。
- `coco_syn_yolox_tiny.py`：COCO SynRAW + `COCO_Syn_preprocess` / `COCO_R_Net_taisp` 布局；`COCORawDataset(image_backend="jpeg")` + `TrainTransformRaw` + TA-YOLOX。

数据管线说明：`COCODataset` + `TrainTransform`（uint8 → ~0–255 float）与 `COCORawDataset` + `TrainTransformRaw`（~0–1，gzip 或 jpeg）并存。

### 训练 / 评估（YOLOX）

在 **`mmdetection_github` 根目录**执行（脚本内会 `cd` 到仓库根并设置 `PYTHONPATH`）：

```bash
bash train.sh
bash eval.sh
```

不传 `-f` 时，`main.py` / `eval.py` 默认使用本仓库内的 **`lod_yolox_tiny.py`**（绝对路径，不依赖当前工作目录）。

**常用环境变量**（见 `train.sh` / `eval.sh` 文件头注释）：

| 变量 | 含义 |
|------|------|
| `YOLOX_EXP` | 实验文件，默认 `lod_yolox_tiny.py`；COCO 版 LOD 可设为 `lod_coco_format_yolox_tiny.py` |
| `EXPN_NAME` | 输出目录名，默认与 `YOLOX_EXP` 主文件名一致；训练与 eval 需一致 |
| `BATCH_SIZE` / `NUM_GPUS` / `PRETRAIN_CKPT` / `CKPT` / `CUDA_VISIBLE_DEVICES` | 按需覆盖 |

示例：

```bash
YOLOX_EXP=lod_coco_format_yolox_tiny.py EXPN_NAME=lod_coco_run bash train.sh
```

