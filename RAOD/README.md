# TA-ISP Object Detection (ROD dataset)

### Dataset Downloads
Download ROD dataset  from [百度网盘 (passwd: udqg)](https://pan.baidu.com/s/1aBroLAj_0XeyQSzJdKrJ7g?pwd=udqg). The dataset is randomly partitioned into a 9:1 train–validation split for both the day and night subsets using the following script:`./scripts/split_rod.py`

Unzip it under any folder, which format as:

```
-- ROD_TAISP
    -- annotations (detection label)
        --train_day640.json
        --train_night640.json
        --valid_day640.json
        --valid_night640.json
    -- day_raws_debayer_fp32_640x640
        -- Day
    -- night_raws_debayer_fp32_640x640
        -- Night
```
You can also download the original dataset (Only training dataset) from [DIAP](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US) and process the dataset using the script in `./scripts/process_raw.py` and `./scripts/process_anno.py`.


### Requirements
Codes are tested under python 3.9.19, torch 2.1.0, torchvision 0.16.0.

### Preprocessing
As mentioned in the paper, we use the pretrained weights of YOLOX to initialize our model. Download the checkpoint that matches your experiment file (see `./cfg/` below):

- **YOLOX-Tiny** (default in `train.sh`): [yolox_tiny.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth)
- **YOLOX-S** (when using `cfg/cfg_small.py` or `cfg/cfg_lod_small.py`): [yolox_s.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)

Place the file under `./pre-trained/`.

### Training configs (`./cfg/`)
All experiment descriptions live under [`./cfg/`](./cfg/). Run `main.py` from the `RAOD` directory so paths like `cfg/cfg_tiny.py` resolve correctly.

| File | Dataset | Backbone |
|------|---------|----------|
| `cfg/cfg_tiny.py` | ROD (COCO raw) | YOLOX-Tiny (`width=0.375`) |
| `cfg/cfg_small.py` | ROD (COCO raw) | YOLOX-S (`width=0.50`) |
| `cfg/cfg_lod_tiny.py` | LOD (VOC-style) | YOLOX-Tiny |
| `cfg/cfg_lod_small.py` | LOD (VOC-style) | YOLOX-S |

Example:

```bash
# ROD + Tiny (paper default)
python main.py -f cfg/cfg_tiny.py -b 4 -c pre-trained/yolox_tiny.pth

# LOD + YOLOX-S
python main.py -f cfg/cfg_lod_small.py -b 48 -c pre-trained/yolox_s.pth
```

The provided `train.sh` uses LOD + Tiny (`cfg/cfg_lod_tiny.py`). Edit `-f` / `-c` there to switch dataset or backbone.

### Training
After all these preprocessing, you can run:

```
bash train.sh
```

### Evaluation
To evaluate the model's performance, please follow these steps:

#### 1.Download Pretrained Weights:

Download our pretrained models from [releases](https://github.com/CVL-UESTC/TA-ISP/releases/tag/v0.0).

#### 2.Run Evaluation:

Execute the evaluation script by running the following command in your terminal (adjust `-f` to match the config used for training):

```
bash eval.sh
```

`eval.sh` defaults to `cfg/cfg_tiny.py` for ROD evaluation.




## Benchmark

#### Standard Models.

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

</details>

#### Light Models.

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |


<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.3  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.pth) |

</details>