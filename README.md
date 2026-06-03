<p align="center">
    <img src="assets/lae-dino.png" alt="Image" width="70">
</p>
<div align="center">
<h1 align="center">Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community</h1>

<h4 align="center"><em>Jiancheng Pan*,     Yanxing Liu*,     Yuqian Fu✉,     Muyuan Ma,</em></h4>

<h4 align="center"><em>Jiahao Li,     Danda Pani Paudel,    Luc Van Gool,     Xiaomeng Huang✉ </em></h4>
<p align="center">
    <img src="assets/inst.png" alt="Image" width="500">
</p>

\* *Equal Contribution* &nbsp; &nbsp; Corresponding Author ✉

</div>

<p align="center">
    <a href="http://arxiv.org/abs/2408.09110"><img src="https://img.shields.io/badge/Arxiv-2408.09110-b31b1b.svg?logo=arXiv"></a>
    <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32672"><img src="https://img.shields.io/badge/AAAI'25-Paper-blue"></a>
    <a href="https://jianchengpan.space/projects/LAE/"><img src="https://img.shields.io/badge/LAE-Project_Page-<color>"></a>
    <a href="https://huggingface.co/datasets/jaychempan/LAE-1M"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-yellow?style=flat&logo=hug"></a>
    <a href="https://github.com/jaychempan/LAE-DINO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow"></a>
</p>

<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#engine">Engine</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#model">Model</a> |
  <a href="#statement">Statement</a>
</p>

<!-- ## TODO

- [X] Release LAE-Label Engine
- [X] Release LAE-1M Dataset
- [ ] Release LAE-DINO Model -->

## News
- [2026/6/3] Thanks to ModelScope's computing support, LAE-DINO has been deployed on [ModelScope Studio](https://modelscope.cn/studios/ML4Sustain/LAE-DINO).
- [2025/8/21] All our LAE-1M dataset support is available for download at 🤗 [HuggingFace](https://huggingface.co/datasets/jaychempan/LAE-1M).
- [2025/4/19] We add the inference examples and the original annotation, and the processed annotation file of DIOR and DOTAv2.
- [2025/3/19] We add the LAE-DINO's config fine-tuned on DIOR and DOTAv2.
- [2025/2/28] We have open sourced the <a href="#model">LAE-DINO Model </a>.
- [2025/2/5] We have open sourced the <a href="#dataset">LAE-1M Dataset </a>.
- [2025/2/5] The LAE-80C dataset, containing 80 classes, has been released as a new remote sensing OVD benchmark and can be quickly [downloaded](https://drive.google.com/drive/folders/1HPu97-f1SNF2sWm3Cdb2FHLRybdRbCtS?usp=sharing) here.
- [2025/1/17] We have open sourced the code for <a href="#engine">LAE-Label Engine </a>.
- [2024/12/10] Our paper of "Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community" is accepted [AAAI&#39;25](https://aaai.org/conference/aaai/aaai-25/), we will open source as soon as possible!
- [2024/8/17] Our paper of "Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community" is up on [arXiv](http://arxiv.org/abs/2408.09110).

## Abstract

Object detection, particularly open-vocabulary object detection, plays a crucial role in Earth sciences, such as environmental monitoring, natural disaster assessment, and land-use planning. However, existing open-vocabulary detectors, primarily trained on natural-world images, struggle to generalize to remote sensing images due to a significant data domain gap. Thus, this paper aims to advance the development of open-vocabulary object detection in remote sensing community. To achieve this, we first reformulate the task as Locate Anything on Earth (LAE) with the goal of detecting any novel concepts on Earth. We then developed the LAE-Label Engine which collects, auto-annotates, and unifies up to 10 remote sensing datasets creating the LAE-1M - the first large-scale remote sensing object detection dataset with broad category coverage. Using the LAE-1M, we further propose and train the novel LAE-DINO Model, the first open-vocabulary foundation object detector for the LAE task, featuring Dynamic Vocabulary Construction (DVC) and Visual-Guided Text Prompt Learning (VisGT) modules. DVC dynamically constructs vocabulary for each training batch, while VisGT maps visual features to semantic space, enhancing text features. We comprehensively conduct experiments on established remote sensing benchmark DIOR, DOTAv2.0, as well as our newly introduced 80-class LAE-80C benchmark. Results demonstrate the advantages of the LAE-1M dataset and the effectiveness of the LAE-DINO method.

<p align="center">
    <img src="assets/lae.png" alt="Image" width="500">
</p>

## Engine

### LAE-Label Engine Pipeline

The pipeline of our LAE-Label Engine. For LAE-FOD dataset, we use coco slice of open-source tools [SAHI](https://github.com/obss/sahi) to automatically slice COCO annotation and image files ([coco-slice-command-usage](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)). For LAE-COD dataset, we build it with the following series of commands (<a href="###how-to-use-lae-label">How to use LAE-Label </a>). We uniformly convert to COCO format. You can refer to my environment [lae_requirements.txt](http://vgithub.com/jaychempan/LAE-DINO/blob/main/internvl_requirements.txt) if you encounter problems.

<p align="center">
    <img src="assets/LAE-Label.png" alt="Image" width="500">
</p>

### How to use LAE-Label
LAE-Label is mainly based on the [SAM](https://github.com/facebookresearch/segment-anything) and [InternVL](https://github.com/OpenGVLab/InternVL/tree/main) projects, mainly referring to the InternVL environment installation.

**Note:** transformers==4.42.3 or 4.45.2(InternVL maybe not install higher transformers version), maybe you can ref the `./internvl_requirements.txt` for installing internvl enviroment.

(Optional) For high resolution remote sensing images, we crop to `1024x1024` size,

```
python LAE-Label/crop_huge_images.py --input_folder ./LAE_data/DATASET --output_folder ./LAE_data/DATASET_sub
```

SAM is then used to obtain the region of interst (RoI) of the image,

```
python LAE-Label/det_with_sam.py --checkpoint ./models/sam_vit_h_4b8939.pth --model-type 'vit_h' --input path/to/images/ --output ./LAE_data/DATASET_sub/seg/ --points-per-side 32 --pred-iou-thresh 0.86 --stability-score-thresh 0.92 --crop-n-layers 1 --crop-n-points-downscale-factor 2 --min-mask-region-area 10000
```

Then crop to get the RoI,

```
python LAE-Label/crop_rois_from_images.py --img_dir ./LAE_data/DATASET_sub/ --base_dir ./LAE_data/DATASET_sub/seg/ --out_dir ./LAE_data/DATASET_sub/crop/ --N 10 --end jpg
```

The currently used current open source model with the best multimodal macromodel results, InternVL, provides two versions, the front is the basic version, but the weight is too large, the latter model with `16% of the model size, 90% of the performance`.

```
huggingface-cli download --resume-download OpenGVLab/InternVL-Chat-V1-5 --local-dir InternVL-Chat-V1-5
huggingface-cli download --resume-download OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir Mini-InternVL-Chat-4B-V1-5
```

We also tested InternVL models of different sizes, including InternVL2-8B (16 GB), InternVL-Chat-V1-5 (48 GB), and InternVL2-26B (48 GB).
<p align="center">
    <img src="assets/LAE-Engine-Test.png" alt="Image" width="800">
</p>

Use the LVLM model to generate the corresponding RoI categories according to the prompt template. It can be deployed locally and remotely, with specific code below. Where remote deployment can be found in [lmdeploy](https://github.com/InternLM/lmdeploy).

```
# local inference
python LAE-Label/internvl-infer.py --model_path ./models/InternVL-Chat-V1-5 --root_directory ./LAE_data/DATASET_sub/crop --csv_save_path ./LAE_data/DATASET_sub/csv/

# remote inference

python LAE-Label/internvl-infer-openai.py --api_key OPENAIAPIKEY --base_url https://oneapi.XXXX.site:8888/v1  --model_name "internvl-internlm2" --input_dir ./LAE_data/DATASET_sub/crop --output_dir ./LAE_data/DATASET_sub/csv/
```

(Optional) Then convert to [odvg dataset format](https://github.com/longzw1997/Open-GroundingDino/blob/main/data_format.md) for better post-processing and other operations,

```
python LAE-Label/dets2odvg.py
```

(Optional) If you want to see the RoI visualization, by querying the image in odvg format. Converted to ODVG for easy visualization and processing.

```
python LAE-Label/plot_bboxs_odvg_dir.py
```

(Optional) Further optimise the quality of the labelling and develop some rules,  refer [post process method](./LAE-Label/post_process/README.md).

Some examples of labelling using LAE-Label, but without rule-based filtering operations.

<p align="center">
    <img src="assets/LAE-Label-PIC.png" alt="Image" width="700">
</p>

## Dataset

LAE-1M dataset contains abundance categories composed of coarse-grained LAE-COD and fine-grained LAE-FOD. LAE-1M samples from these datasets by category and does not count instances of overlap duplicates when slicing.

<p align="center">
    <img src="assets/LAE-1M.png" alt="Image" width="700">
</p>

### Dowload LAE-1M Dataset

Download data can be downloaded through `Baidu disk` or `Onedrive`, the download address provided below is downloaded to the `./LAE-DINO` of the project.

Note: **LAE-Label Engine is continuously optimized, the quality of data annotation is also improved.** We try to explore higher quality data annotations, and dataset versions are iteratively updated. The current version dataset is v1.1, which is the best labelled version available. We also intend to build stable benchmarks based on this version.

> Baidu disk: [download link](https://pan.baidu.com/s/1_l2i0gUPcDbTUkNkUqEhjg?pwd=chrx)

> Onedrive: [download link](https://1drv.ms/f/c/72d4076f2aa319be/EhpYDEA71mFOorBWIoxglwMBNuy3i3bbf2W1qi8IHBjOAA?e=mGThPR)

Once you have downloaded the dataset, you can extract the image files in all subdirectories with a shell command.

```
bash tools/unzip_all_images_files.sh
```

We have preserved the image catalog names of the original datasets (e.g. DOTA,DIOR et.al.) as much as possible, so it is possible to incrementally download parts (SLM, EMS) of the image data, and separate labeled files.

e.g. Extract `images` from origin datasets:

```
# DOTAv2 dataset
cd DOTAv2/
unzip images.zip

# DIOR dataset
cd DIOR/
unzip images.zip

# FAIR1M dataset
cd FAIR1M/
unzip images.zip

# NWPU-VHR-10 dataset
cd NWPU-VHR-10/
unzip images.zip


# HRSC2016 dataset
cd HRSC2016/
unrar x HRSC2016.part01.rar
mv Train/AllImages ../Train/AllImages

# RSOD dataset:
cd RSOD/
mkdir ../images
mv aircraft/JPEGImages ../images
mv oiltank/JPEGImages ../images
mv overpass/JPEGImages ../images
mv aircraft/JPEGImages ../images
```

### LAE-80C Benchmark

LAE-80C is sampled from the validation set of multiple remote sensing object detection datasets to filter the categories that are as semantically non-overlapping as possible. We combined these categories to create a benchmark with 80 categories.

<p align="center">
    <img src="assets/LAE-80C.png" alt="Image" width="700">
</p>

**There is a lack of larger categories of detection benchmarks for the remote sensing community.** The LAE-80C can be used alone as a standard for evaluating 80-class object detection in remote sensing scenarios. Here is a quick [download](https://drive.google.com/drive/folders/1HPu97-f1SNF2sWm3Cdb2FHLRybdRbCtS?usp=sharing) via google drive.

### Dataset Catalogue

The directory structure of the `./data` file is shown below. In order to unify the various structures, we can directly use the coco format data. `Power-Plant` is the `Condesing-Towering` of paper.

```
./data
├── LAE-80C
│   ├── images
│   ├── LAE-80C-benchmark_categories.json
│   ├── LAE-80C-benchmark.json
│   └── LAE-80C-benchmark.txt
├── LAE-COD
│   ├── AID
│   ├── EMS
│   ├── NWPU-RESISC45
│   └── SLM
└── LAE-FOD
    ├── DIOR
    ├── DOTAv2
    ├── FAIR1M
    ├── HRSC2016
    ├── NWPU-VHR-10
    ├── Power-Plant
    ├── RSOD
    └── Xview
```

## Model

The pipeline for solving the LAE task: LAE-Label Engine expands vocabulary for open-vocabulary pre-training; LAE-DINO is a DINO-based open-vocabulary detector with Dynamic Vocabulary Construction (DVC) and Visual-Guided Text Prompt Learning (VisGT), which has a pre-training and fine-tuning paradigm for open-set and closed-set detection.

<p align="center">
    <img src="assets/LAE-DINO-Pipeline.png" alt="Image" width="700">
</p>

### Installation Environment

The experimental environment is based on [mmdetection](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md), the installation environment reference mmdetection's [installation guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md). You can refer to my environment [lae_requirements.txt](http://vgithub.com/jaychempan/LAE-DINO/blob/main/lae_requirements.txt) if you encounter problems.
```
conda create --name lae python=3.8 -y
conda activate lae
cd LAE-DINO/mmdetection_lae
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# 开发并直接运行 mmdet
pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
Then download the BERT weights `bert-base-uncased` into the weights directory,
```
cd LAE-DINO
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir weights/bert-base-uncased
```

### Train LAE-DINO Model

#### Pre-training

```
./tools/dist_train.sh configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py 4
```

Continuing training from the last training breakpoint,

```
./tools/dist_train_lae.sh configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py 4
```

#### Fine-tuning

```
# DIOR
./tools/dist_train.sh configs/lae_dino/lae_dino_swin-t_finetune_DIOR.py 4
# DOTAv2
./tools/dist_train.sh configs/lae_dino/lae_dino_swin-t_finetune_DOTA.py 4
```

**Note: About Data Labeling Description**

| File Name                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| DOTAv2_train.json                | Original annotation file                                                   |
| processed_DOTAv2_train.json      | Splits images with too many annotations to prevent GPU memory overflow     |
| processed_LAE-1M_DOTAv2_train.json | Sampled from `processed_DOTAv2_train.json` to construct the LAE-1M dataset |

### Test LAE-DINO Model

```
./tools/dist_test.sh configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py /path/to/model/ 4
```

### Inference LAE-DINO Model

#### Open Vocabulary Object Detection

For one image example:
```
python demo/image_demo.py images/airplane.jpg \
        configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py \
        --weights /path/to/model/ \
       --texts 'playground . road . tank . airplane . vehicle' -c \
       --palette random \
       --pred-score-thr 0.4
```
For one dirs:
```
python demo/image_demo.py images/ \
        configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py \
        --weights /path/to/model/ \
       --texts 'playground . road . tank . airplane . vehicle' -c \
       --palette random \
       --pred-score-thr 0.4
```

### Analyze

```
python tools/analysis_tools/analyze_logs.py plot_curve 
```

Based on the stable version of the LAE-1M dataset, we used 4-card A100 and ran 32 epochs with 4 per card batch size. LAE-80C considers more categories and can be used as a benchmark for zero-shot and few-shot in remote sensing.

| Method      | Backbone      | Pre-Train Data      | DIOR AP50 | DOTAv2.0 mAP | LAE-80C mAP | Weights |
|------------|------------|------------|------|-------------|-------------|-------------|
| LAE-DINO | Swin-T |LAE-1M | 87.3 | 51.5        | 24.1  | [[weight]](https://drive.google.com/file/d/1EiR8KtNRYIeOfvtIe9C82cQk_uOMIQ8U/view?usp=sharing)
| LAE-DINO-FT | Swin-T|- | 92.0 | -        | -  | [[weight]](https://drive.google.com/file/d/1Q35PUdzUHLIM22ozm_ccgJzXXtfMXSOm/view?usp=sharing)
| LAE-DINO-FT | Swin-T|- | - | 55.5        | - | [[weight]](https://drive.google.com/file/d/1rxDO2QGNWw8WiDKRTqqVrK-JE4rE3VVT/view?usp=sharing)

## Discussion
- Our work is suitable for zero-shot and few-shot benchmark models in remote sensing, which can be used for pre-detection of some common and uncommon categories.

- Regarding “Locate” and “Detect”, in most of the articles, these two are the same, because most of the tasks are only concerned with the relative position of the ROI, but in the actual remote sensing detection, the latitude and longitude can be calculated by calculating the world position  and the position of the image.

## Statement

### Acknowledgement

This project references and uses the following open source models and datasets. Thanks also to `ETH Zürich` and `INSAIT` for partial computing support.

#### Related Open Source Models

- [MM-Grounding-DINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)
- [segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file)
- [InternVL](https://github.com/OpenGVLab/InternVL/tree/main)
- [MTP](https://github.com/ViTAE-Transformer/MTP)

#### Related Open Source Datasets

- [DOTA Dataset](https://captain-whu.github.io/DOTA/dataset.html)
- [DIOR Dataset](http://www.escience.cn/people/gongcheng/DIOR.html)
- [FAIR1M Dataset](https://arxiv.org/abs/2103.05569)
- [AID Dataset](https://captain-whu.github.io/AID/)
- [RSICD Dataset](https://github.com/201528014227051/RSICD_optimal)
- [NWPU Dataset](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)
- [RSOD Dataset](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-)
- [HRSC2016 Dataset](http://www.escience.cn/people/liuzikun/DataSet.html)
- [Power-Plant Dataset](https://github.com/SPDQ/Power-Plant-Detection-in-RSI)
- [SLM Dataset](https://github.com/xiaoyuan1996/SemanticLocalizationMetrics)

### Citation

If you are interested in the following work, please cite the following paper.

```
@inproceedings{pan2025locate,
  title={Locate anything on earth: Advancing open-vocabulary object detection for remote sensing community},
  author={Pan, Jiancheng and Liu, Yanxing and Fu, Yuqian and Ma, Muyuan and Li, Jiahao and Paudel, Danda Pani and Van Gool, Luc and Huang, Xiaomeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6281--6289},
  year={2025}
}

or

@misc{pan2024locateearthadvancingopenvocabulary,
    title={Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community}, 
    author={Jiancheng Pan and Yanxing Liu and Yuqian Fu and Muyuan Ma and Jiaohao Li and Danda Pani Paudel and Luc Van Gool and Xiaomeng Huang},
    year={2024},
    eprint={2408.09110},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2408.09110}, 
}
```
