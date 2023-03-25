# DSTNet

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/xuboming8/CDVD-TSPNL/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10.1-%237732a8)](https://pytorch.org/)

### Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring
By [Jinshan Pan*](https://jspan.github.io/), Boming Xu*, Jiangxin Dong,  Jianjun Ge and Jinhui Tang

> **Abstract**: *How to effectively explore spatial and temporal information is important for video deblurring. In contrast to existing methods that directly align adjacent frames without discrimination, we develop a deep discriminative spatial and temporal network to facilitate the spatial and temporal feature exploration for better video deblurring. We first develop a channel-wise gated dynamic network to adaptively explore the spatial information. As adjacent frames usually contain different contents, directly stacking features of adjacent frames without discrimination may affect the latent clear frame restoration. Therefore, we develop a simple yet effective discriminative temporal feature fusion module to obtain useful temporal features for latent frame restoration. Moreover, to utilize the information from long-range frames, we develop a wavelet-based feature propagation method that takes the discriminative temporal feature fusion module as the basic unit to effectively propagate main structures from long-range frames for better video deblurring. We show that the proposed method does not require additional alignment methods and performs favorably against state-of-the-art ones on benchmark datasets in terms of accuracy and model complexity*


This repository is the official PyTorch implementation of "Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring"

## Network Architecture
[![ppDnq0A.png](https://s1.ax1x.com/2023/03/25/ppDnq0A.png)](https://imgse.com/i/ppDnq0A)

## Updates
[2022-02-28] Paper has been accepted by CVPR2023\
[2023-03-25] Training & Testing code is available!

## Experimental Results

 
