#!/bin/bash
set -e  # 遇到错误时退出

echo "=== Step 1: 安装 PyTorch + CUDA 12.1 ==="
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "=== Step 2: 安装 face_verification requirements ==="
pip install -r ./face_verification/requirements.txt
pip install facenet_pytorch --no-deps

echo "=== Step 3: 安装 content_moderation requirements ==="
pip install -r ./content_moderation/requirements.txt

echo "✅ 全部安装完成"
