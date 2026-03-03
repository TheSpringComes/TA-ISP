# TA-ISP Semantic Segmentation (ADE20K-RAW dataset)

### 📖 1: Dataset Download

**ADE20K-RAW** is a synthesised RAW data semantic segmentation dataset, we follow [RAW-Adapter](https://github.com/cuiziteng/ECCV_RAW_Adapter) and adopt InverseISP to translate RGB to raw-RGB data, then apply inverse white balance & mosaic on the translated data.

Download the dataset from [Google Drive](https://drive.google.com/file/d/1OZ4_rbJqlmlvmIjWzM5J4JjQCF2-fatP/view?usp=sharing) or [百度网盘 (passwd: acv7)](https://pan.baidu.com/s/1hv4Dc6AGBRr1u-7OgJ0zfA)

Dataset Format as:

```
--  data
     -- ADE20K
         -- ADEChallengeData2016
             
             -- annotations
                 -- training
                 -- validation
             
             -- images

                 # Synthesis RAW data
                 -- training_raw (synthesis RAW)
                 -- validation_raw
                 -- training_raw_low  (synthesis RAW, low-light)
                 -- validation_raw_low
                 -- training_raw_over_exp  (synthesis RAW, over-exposure)
                 -- validation_raw_over_exp

                 # Compare ISP methods
                  -- ISP_methods
                      -- 1_SID (https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf)
                      -- 2_InvISP (https://yzxing87.github.io/InvISP/index.html)
                      -- 3_ECCV16(https://karaimer.github.io/camera-pipeline/)
```

### 📖 2: Enviroment Setup

Our code based on [mmsegmentation 1.2.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1) version, you can following their [instruction](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.1/docs/en/get_started.md#installation) to build environment. Or following our steps below:

(1). Create a conda environment and activate it:
```
conda create --name mmseg python=3.8 -y
conda activate mmseg
```

(2). Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. ours (torch1.11.0+cu113):
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

(3). Mmsegmentation setup:
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Develop mmseg:
```
pip install -v -e .
```


### 📖 3: Model Evaluation 


We show the **mIOU** performance. Download the pretrain weights (ckpt) from [release](https://github.com/CVL-UESTC/TA-ISP/releases).

**TA-ISP** (Segformer - MITB0 backbone): 

| Light Condition | Normal-Light | Low-Light | 
|  ---- | ---- | ---- |
| TA-ISP (MITB0) | 36.29 / [config](configs/CVPR26_TAISP/taisp_normal_mitb0.py) [[ckpt](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.1/ade20k_normal_taisp.pth)] | 26.77 / [config](configs/CVPR26_TAISP/taisp_low_mitb0.py) [[ckpt](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.1/ade20k_low_taisp.pth)]


Evaluation of TA-ISP, only need single GPU (low-light, MIT-B0 backbone for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/CVPR26_TAISP/taisp_low_mitb0.py https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.1/ade20k_low_taisp.pth
```

### 📖 4: Model Training (Optional) 

```
sh tools/train.sh
```



### 📖 Acknowledgement:

We thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1) & [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) for their excellent code base & dataset, and [InvISP](https://yzxing87.github.io/InvISP/index.html) for the RAW data synthesis contribution.





