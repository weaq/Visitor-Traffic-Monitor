#!/bin/bash

set -e

echo "=== [1/10] ปรับ Jetson เป็นโหมดเต็มกำลัง ==="
sudo nvpmodel -m 0
sudo jetson_clocks

echo "=== [2/10] ลบโปรแกรมไม่จำเป็น ==="
sudo apt-get purge -y libreoffice*
sudo apt-get clean
sudo apt-get update && sudo apt-get upgrade -y

echo "=== [3/10] ติดตั้ง dependencies สำหรับ build OpenCV และระบบ ==="
sudo apt-get install -y \
    git cmake gfortran nano locate wget unzip \
    libatlas-base-dev libhdf5-serial-dev hdf5-tools \
    python3-dev build-essential pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libavresample-dev \
    libtbb2 libtbb-dev libtiff-dev libjpeg-dev \
    libpng-dev libdc1394-22-dev libgtk-3-dev \
    libcanberra-gtk3-module

echo "=== [4/10] ติดตั้ง Python 3.8 และ venv ==="
sudo apt-get install -y python3.8 python3.8-dev python3.8-venv
python3.8 -m venv ~/yoloenv
source ~/yoloenv/bin/activate

echo "=== [5/10] อัปเกรด pip และ setuptools (จำกัดเวอร์ชัน) ==="
pip install --upgrade pip
pip install "setuptools<70"

echo "=== [6/10] คอมไพล์ OpenCV 4.5.2 พร้อม CUDA ==="
cd ~
wget https://github.com/opencv/opencv/archive/4.5.2.zip -O opencv-4.5.2.zip
wget https://github.com/opencv/opencv_contrib/archive/4.5.2.zip -O opencv_contrib-4.5.2.zip
unzip opencv-4.5.2.zip
unzip opencv_contrib-4.5.2.zip
mkdir -p opencv-4.5.2/build
cd opencv-4.5.2/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="5.3,6.2,7.2" \
      -D WITH_CUBLAS=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D WITH_GSTREAMER=OFF \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.2/modules \
      ..
make -j4
sudo make install
sudo ldconfig

echo "=== [7/10] ติดตั้ง NumPy (รองรับ OpenCV) ==="
pip install numpy

echo "=== [8/10] ติดตั้ง PyTorch 1.13.0 (Python 3.8, Jetson Nano) ==="
cd ~
wget https://nvidia.box.com/shared/static/p57jwnt0a0fylxh3aj3zvylp1fuzx6il.whl -O torch-1.13.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0-cp38-cp38-linux_aarch64.whl

echo "=== [9/10] ติดตั้ง YOLOv8 จาก GitHub ==="
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
cd ~

echo "=== [10/10] ติดตั้ง DeepSort ==="
pip install deep_sort_realtime

echo "🎉 เสร็จสิ้น! ใช้งาน YOLOv8 ได้แล้ว ✅"
echo "👉 ใช้ virtualenv ด้วยคำสั่ง: source ~/yoloenv/bin/activate"
