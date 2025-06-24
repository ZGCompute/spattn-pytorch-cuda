# SpAttn for SAR Imagery – PyTorch + CUDA

This repository contains an optimized sparse attention module (`SpAttn`) for processing high-resolution Synthetic Aperture Radar (SAR) imagery. The implementation is based on the 2021 HPCA paper by Song Han et al., extended with PyTorch C++/CUDA custom extensions for efficient inference and training.

## 🚀 Features

- Sparse attention operator accelerated via custom CUDA kernels.
- Memory-efficient inference for large SAR data volumes.
- Training script and evaluation pipeline with test SAR imagery.
- Designed for use in low-latency or edge-compute scenarios.

## 📦 Installation

```bash
git clone https://github.com/yourusername/sar-spattn-pytorch-cuda.git
cd sar-spattn-pytorch-cuda
pip install -r requirements.txt
python extensions/build.py install

## Usage

from models.spattn import SpAttn
import torch

x = torch.load("data/sample_sar_input.pt").cuda()  # shape: [B, C, H, W]
attn = SpAttn(dim=256, heads=4, sparsity=0.7).cuda()
y = attn(x)

## Run Tests

pytest tests/

## Citation

If you use this code, please cite the original HPCA 2021 paper.
https://arxiv.org/abs/2102.00586

