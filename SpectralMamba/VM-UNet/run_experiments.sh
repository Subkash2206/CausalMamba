#!/bin/bash
# Using absolute path to the vmunet environment's python binary
PYTHON_BIN="/root/miniconda/envs/vmunet/bin/python"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Starting UNet-ResNet50 Training..."
echo "=========================================="
$PYTHON_BIN -u train_unet_isic18.py | tee unet_isic18_train.log

echo "=========================================="
echo "UNet-ResNet50 Training Complete."
echo "Starting Swin-UNet Training..."
echo "=========================================="
$PYTHON_BIN -u train_swinunet_isic18.py | tee swinunet_isic18_train.log

echo "=========================================="
echo "Swin-UNet Training Complete."
echo "Starting VM-UNet Training (Scratch)..."
echo "=========================================="
$PYTHON_BIN -u train_vmunet_isic18.py | tee vmunet_scratch_isic18_train.log

echo "=========================================="
echo "All Training Complete."
echo "=========================================="
