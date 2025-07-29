#!/bin/bash
set -e

echo "🔧 Starting YOLOv8 setup for Jetson Nano..."

# === STEP 1: System Update ===
echo "🔄 Updating system..."
sudo apt update && sudo apt upgrade -y

# === STEP 2: Python 3.6 Virtual Environment ===
echo "🐍 Setting up Python 3.6 virtual environment..."
sudo apt install -y python3-pip python3-virtualenv
virtualenv -p python3 venv_yolo
source venv_yolo/bin/activate

# === STEP 3: Install Required Python Packages ===
echo "📦 Installing Python packages..."

# Install numpy first to avoid build issue
pip install --upgrade pip
pip install numpy==1.19.5

# === STEP 4: Install PyTorch 1.10.0 (with CUDA 10.2 support) ===
echo "⚙️ Installing PyTorch 1.10.0 for JetPack 4.4 - 4.6..."

# Download wheel
wget https://nvidia.box.com/shared/static/2sj9kxz8e3vnzsdn5dfzhd0a7ogq0fl3.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install it
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# === STEP 5: Install Ultralytics YOLOv8 ===
echo "🧠 Installing YOLOv8..."
pip install opencv-python==4.5.5.64
pip install ultralytics==8.0.20

# === STEP 6: Test Torch CUDA ===
echo "🧪 Verifying PyTorch and CUDA..."
python3 -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# === STEP 7: Download YOLOv8 model (nano) ===
echo "⬇️ Downloading YOLOv8n model..."
yolo task=detect mode=predict model=yolov8n.pt imgsz=640 source='https://ultralytics.com/images/bus.jpg' save=True

echo "✅ YOLOv8 setup completed successfully!"
echo "🚀 Run example: source venv_yolo/bin/activate && yolo task=detect mode=predict model=yolov8n.pt source=0"
