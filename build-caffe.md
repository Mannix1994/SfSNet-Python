# Build caffe on Ubuntu

## 1. Install denpendencies
```bash
sudo apt-get update 
sudo apt-get install --yes cmake git build-essential
sudo apt-get install --yes libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --yes --no-install-recommends libboost-all-dev
sudo apt-get install --yes libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install --yes libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install --yes python-pip
```

## 2. Clone caffe from github
```bash
git clone https://github.com/BVLC/caffe
cd caffe
```

## 3. Install pycaffe denpendencies
modify matplotlib>=1.3.1 in python/requirements.txt to 
matplotlib==2.2.3  
modify ipython>=3.0.0 in python/requirements.txt to 
ipython==5.0.0

```bash
pip install -r python/requirements.txt
```

## 4. Build caffe
run cmake command
```bash
mkdir cmake-build
cd cmake-build
cmake -D CMAKE_INSTALL_PREFIX=/opt/caffe ..
```
CMAKE_INSTALL_PREFIX specify the install directory, caffe will
be install to /opt/caffe. cmake's output is like:
```
-- ******************* Caffe Configuration Summary *******************
-- General:
--   Version           :   1.0.0
--   Git               :   1.0-132-g99bd997-dirty
--   System            :   Linux
--   C++ compiler      :   /usr/bin/c++
--   Release CXX flags :   -O3 -DNDEBUG -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
--   Debug CXX flags   :   -g -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
--   Build type        :   Release
-- 
--   BUILD_SHARED_LIBS :   ON
--   BUILD_python      :   ON
--   BUILD_matlab      :   OFF
--   BUILD_docs        :   ON
--   CPU_ONLY          :   OFF
--   USE_OPENCV        :   ON
--   USE_LEVELDB       :   ON
--   USE_LMDB          :   ON
--   USE_NCCL          :   OFF
--   ALLOW_LMDB_NOLOCK :   OFF
--   USE_HDF5          :   ON
-- 
-- Dependencies:
--   BLAS              :   Yes (Atlas)
--   Boost             :   Yes (ver. 1.58)
--   glog              :   Yes
--   gflags            :   Yes
--   protobuf          :   Yes (ver. 2.6.1)
--   lmdb              :   Yes (ver. 0.9.17)
--   LevelDB           :   Yes (ver. 1.18)
--   Snappy            :   Yes (ver. 1.1.3)
--   OpenCV            :   Yes (ver. 2.4.9.1)
--   CUDA              :   No
-- 
-- Python:
--   Interpreter       :   /usr/bin/python2.7 (ver. 2.7.12)
--   Libraries         :   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.12)
--   NumPy             :   /home/creator/.local/lib/python2.7/site-packages/numpy/core/include (ver 1.16.1)
-- 
-- Documentaion:
--   Doxygen           :   No
--   config_file       :   
-- 
-- Install:
--   Install path      :   /opt/caffe
```

then:
```bash
# replace 8 with your cpu kernel count
make all -j8
sudo make install
sudo chmod -R 777 /opt/caffe
# make symbolic to python libs
ln -s /opt/caffe/python/caffe ~/.local/lib/python2.7/site-packages/caffe

```
## 5. Test pycaffe
create caffe_test.py, write code:
```python
import sys
import os

import caffe

if __name__ == '__main__':
	caffe.set_mode_cpu()
	print 'You have good luck!'
```
then:
```bash
python caffe_test.py
```
if there is no error message, congratulations, you made it