#!/bin/bash

set -e

echo "📦 อัปเดตระบบ..."
sudo apt update
sudo apt install -y python3-pip python3-dev python3-venv libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev

echo "🧱 สร้าง virtual environment..."
python3 -m venv yolov8_env
source yolov8_env/bin/activate

echo "⬆️ อัปเกรด pip และ setuptools..."
pip install --upgrade pip==21.3.1 setuptools==59.6.0

echo "🔥 ติดตั้ง PyTorch 1.10.0 (สำหรับ JetPack 4.4–4.6)..."
wget https://nvidia.box.com/shared/static/8s03e41vld9vypm6rfxu9f19faw0gqj0.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

echo "🧠 ติดตั้ง torchvision แบบ source..."
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision.git
cd vision
python setup.py install
cd ..
rm -rf vision

echo "📸 ติดตั้ง OpenCV..."
pip install opencv-python==4.5.5.64

echo "🧠 ติดตั้ง ultralytics==8.0.20 (เวอร์ชันสุดท้ายที่รองรับ Python 3.6)..."
pip install ultralytics==8.0.20

echo "✅ เสร็จแล้ว! ให้ activate ด้วยคำสั่ง:"
echo "source yolov8_env/bin/activate"
