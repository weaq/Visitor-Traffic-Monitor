#!/bin/bash

echo "===== เพิ่ม Swap 4GB ====="
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "===== ติดตั้ง Python3 + Virtualenv ====="
sudo apt-get update
sudo apt-get install -y python3-pip python3-virtualenv libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev git

echo "===== สร้าง virtualenv สำหรับ yolo_env ====="
virtualenv -p python3.6 yolo_env
source yolo_env/bin/activate

echo "===== ติดตั้ง Cython และ numpy ====="
pip install --upgrade pip
pip install 'Cython<3' numpy

echo "===== ดาวน์โหลดและติดตั้ง PyTorch 1.10.0 ====="
mkdir -p torch_install && cd torch_install
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
cd ..

echo "===== ดาวน์โหลดและติดตั้ง torchvision จาก source ====="
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
pip install -e .
cd ..

echo "===== ติดตั้ง YOLOv8 (ultralytics) รุ่นเบา ====="
pip install ultralytics==8.0.20 opencv-python

echo "===== ทดสอบ PyTorch ====="
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "===== ติดตั้งเสร็จสิ้น ====="
echo "พร้อมใช้งาน YOLOv8n แล้วใน virtualenv 'yolo_env'"
