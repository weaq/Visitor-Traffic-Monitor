# ไปที่โฟลเดอร์สำหรับดาวน์โหลด
mkdir torch_install && cd torch_install

# ดาวน์โหลด PyTorch 1.10.0 ที่รองรับ JetPack 4.6.1 (Python 3.6)
wget https://nvidia.box.com/shared/static/p57jwntvok9467e3cfvk5i2rfsk4g2n2.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# ดาวน์โหลด torchvision ที่เข้ากันได้
wget https://nvidia.box.com/shared/static/f0zs3s7xqfr2eyu6qk8prb5hfz6vgyg1.whl -O torchvision-0.11.1-cp36-cp36m-linux_aarch64.whl
