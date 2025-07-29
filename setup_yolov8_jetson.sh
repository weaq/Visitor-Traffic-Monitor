#!/bin/bash

set -e

echo "✅ 1. Update system"
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev libpython3-dev

echo "✅ 2. Create virtual environment"
python3 -m venv yolov8_env
source yolov8_env/bin/activate

echo "✅ 3. Upgrade pip"
pip install --upgrade pip setuptools wheel

echo "✅ 4. Download PyTorch 1.10.0 for Jetson Nano"
wget https://nvidia.box.com/shared/static/62jymv20x60r61vh3xgxdbr1zjbnj7wu.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

echo "✅ 5. Install PyTorch 1.10.0"
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

echo "✅ 6. Install YOLOv8"
pip install ultralytics==8.0.20

echo "✅ 7. Install OpenCV"
pip install opencv-python==4.5.5.64

echo "✅ 8. Done. Activate environment with:"
echo "     source yolov8_env/bin/activate"
