sudo apt update
sudo apt upgrade
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev

Download the Python source code for version 3.8 from the official Python website. You can use the following command to download it directly to your Jetson Nano:
	wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz


Extract the downloaded archive by running the following command:
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12

Configure the build process:
	./configure --enable-optimizations
	 
Build Python:
make -j4

Once the compilation is complete, you can install Python by running the following command:
	sudo make altinstall
	python3.8 --version

That's it! You have successfully installed Python 3.8 on your Jetson Nano.

Now come out from python3.8 folder and create a separate environment using python 3.8
python3.8 -m venv myenv                                                
source myenv/bin/activate
