# DSTNet

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/xuboming8/CDVD-TSPNL/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10.1-%237732a8)](https://pytorch.org/)

#### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Pan_Deep_Discriminative_Spatial_and_Temporal_Network_for_Efficient_Video_Deblurring_CVPR_2023_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Pan_Deep_Discriminative_Spatial_CVPR_2023_supplemental.pdf) | [Discussion](https://github.com/xuboming8/DSTNet/issues)
### Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring
By [Jinshan Pan*](https://jspan.github.io/), Boming Xu*, [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao),  Jianjun Ge and [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN)

<hr />

> **Abstract**: *How to effectively explore spatial and temporal information is important for video deblurring. In contrast to existing methods that directly align adjacent frames without discrimination, we develop a deep discriminative spatial and temporal network to facilitate the spatial and temporal feature exploration for better video deblurring. We first develop a channel-wise gated dynamic network to adaptively explore the spatial information. As adjacent frames usually contain different contents, directly stacking features of adjacent frames without discrimination may affect the latent clear frame restoration. Therefore, we develop a simple yet effective discriminative temporal feature fusion module to obtain useful temporal features for latent frame restoration. Moreover, to utilize the information from long-range frames, we develop a wavelet-based feature propagation method that takes the discriminative temporal feature fusion module as the basic unit to effectively propagate main structures from long-range frames for better video deblurring. We show that the proposed method does not require additional alignment methods and performs favorably against state-of-the-art ones on benchmark datasets in terms of accuracy and model complexity.*
<hr />


This repository is the official PyTorch implementation of our CVPR2023 paper "Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring".

## Network Architecture
![DSTNet](https://github.com/xuboming8/DSTNet/assets/20449507/d9691c13-9ad9-4d87-846a-f6a9f1bdfb79)

## Updates
[2022-02-28] Paper has been accepted by CVPR2023\
[2023-03-25] Training & Testing code is available!\
[2023-02-04] Extended version [DSTNet+](https://github.com/sunny2109/DSTNet-plus) has been created!

## Experimental Results
Quantitative evaluations on the GoPro dataset. “Ours-L” denotes a large model, where we use 96 features and 30 ResBlocks in the DTFF module.
[![GOPRO](https://s1.ax1x.com/2023/03/25/ppDu8tx.png)](https://imgse.com/i/ppDu8tx)

Quantitative evaluations on the DVD dataset in terms of PSNR and SSIM.
[![DVD](https://s1.ax1x.com/2023/03/25/ppDuGh6.png)](https://imgse.com/i/ppDuGh6)

Quantitative evaluations on the BSD deblurring dataset in terms of PSNR and SSIM.
[![BSD](https://s1.ax1x.com/2023/03/25/ppDut1O.png)](https://imgse.com/i/ppDut1O)

## Dependencies
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.10.1](https://pytorch.org/): `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge`
- Install dependent packages :`pip install -r requirements.txt`
- Install DSTNet :`python setup.py develop`

## Get Started

### Pretrained models
- Models are available in  `'./experiments/model_name'`

### Dataset Organization Form
If you prepare your own dataset, please follow the following form like GOPRO/DVD:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```
 
### Training
- Download training dataset like above form.
- Run the following commands:
```
Single GPU
python basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml
Multi-GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml --launcher pytorch
```

### Testing
- Models are available in  `'./experiments/'`.
- Organize your dataset(GOPRO/DVD/BSD) like the above form.
- Run the following commands:
```
python basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO.yml
cd results
python merge_full.py
python calculate_psnr.py
```
- Before running merge_full.py, you should change the parameters in this file of Line 5,6,7,8.
- The deblured result will be in `'./results/dataset_name/'`.
- Before running calculate_psnr.py, you should change the parameters in this file of Line 5,6.
- We calculate PSNRs/SSIMs by running calculate_psnr.py

## Citation
```
@InProceedings{Pan_2023_CVPR,
    author = {Pan, Jinshan and Xu, Boming and Dong, Jiangxin and Ge, Jianjun and Tang, Jinhui},
    title = {Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
    month = {Feb},
    year = {2023}
}
```
