# Unet
Sementic segmentation in xray dataset

## Introduction

## Architecture

![architecture](images/architecture.png)

**Unet** consists of two parts a contracting side and an expansive side, and create
an elegant U shape architecture. Each side is composed of multiple layers which are built
from 3x3 convolution layers followed by ReLU and a 2x2 max pool. One impressive operation is 
up convolution and to interpret it clearly, I recommend some explanations:
- [Up-sampling with transposed convolution](https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)
- [Understand transposed convolution](https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

## Usage

## Result
