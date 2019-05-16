# Build caffe on Ubuntu

On Ubuntu 18.04, gcc/g++ 6.5 and Matlab 2018a is needed.
install them first. Install gcc6/g++6
```bash
sudo apt-get install gcc-6 g++-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 2 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --config gcc  # choose gcc 6
# make sure is gcc 6 and g++6
gcc -v
g++ -v
```
 On Ubuntu 16.04 gcc/g++ 6.3 and Matlab 2018a is needed. you 
should build gcc/g++ 6.3 first, and then:

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 2 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --config gcc  # choose gcc 6
# make sure is gcc 6 and g++6
gcc -v
g++ -v
```
If you will build caffe with CUDA 10, please update CMAKE to 3.14.3.

## 1. Install denpendencies
```bash
sudo apt-get update 
sudo apt-get install --yes cmake git build-essential
sudo apt-get install --yes libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --yes --no-install-recommends libboost-all-dev
sudo apt-get install --yes libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install --yes libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install --yes python-pip
sudo apt-get install --yes graphviz
```

## 2. Clone caffe from github
```bash
git clone https://github.com/BVLC/caffe
cd caffe
```

## 3. Install pycaffe denpendencies
* modify `matplotlib>=1.3.1` in python/requirements.txt to 
`matplotlib==2.2.4 ` 
* modify `ipython>=3.0.0` in python/requirements.txt to 
`ipython==5.0.0 ` 
* modify `scikit-image>=0.9.3` to `scikit-image==0.14.2`

run command

```bash
pip install pydotplus
pip install -r python/requirements.txt
```

## 4. Build caffe
* Modify cmake configuration  
    * modify line 357 in `cmake/Utils.cmake` from:
        ```txt
          message(FATAL_ERROR "Logic error. Need to update cmake script")
        ```
        to
         ```bash
          # message(FATAL_ERROR "Logic error. Need to update cmake script")
        ```
    * add two lines to `CMakeLists.txt`, from:
        ```txt
        ...
        if(POLICY CMP0054)
          cmake_policy(SET CMP0054 NEW)
        endif()
        ```
        to
        ```txt
        ...
        if(POLICY CMP0054)
          cmake_policy(SET CMP0054 NEW)
        endif()
        include_directories(include)
        include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)  # CUDA include directory
        ```

* Run command:
    ```bash
    mkdir cmake-build
    cd cmake-build
    cmake -D CMAKE_INSTALL_PREFIX=/opt/caffe \
    -D CMAKE_BUILD_TYPE=Release \
    -D OpenCV_DIR=/usr/share/OpenCV \
    -D BUILD_matlab=ON \
    -D Matlab_DIR=/usr/local/MATLAB/R2018a \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 ..
    ```
    CMAKE_INSTALL_PREFIX specify the install directory, caffe will
    be install to /opt/caffe. OpenCV_DIR specify the default 
    OpenCV to use. BUILD_matlab set on to build matcaffe. Matlab_DIR
    specify the root directory of Matlab.
    
    cmake's output is like:
    ```
    ******************* Caffe Configuration Summary *******************
    -- General:
    ...
    -- 
    --   BUILD_SHARED_LIBS :   ON
    --   BUILD_python      :   ON
    --   BUILD_matlab      :   ON
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
    --   Boost             :   Yes (ver. 1.65)
    --   glog              :   Yes
    --   gflags            :   Yes
    --   protobuf          :   Yes (ver. 3.0.0)
    --   lmdb              :   Yes (ver. 0.9.21)
    --   LevelDB           :   Yes (ver. 1.20)
    --   Snappy            :   Yes (ver. ..)
    --   OpenCV            :   Yes (ver. 3.2.0) # 3.2.0 or 2.4.9
    --   CUDA              :   Yes (ver. 9.0)  # can be No, if you don't have cuda
    -- 
    -- NVIDIA CUDA:
    --   Target GPU(s)     :   Auto
    --   GPU arch(s)       :   sm_61
    --   cuDNN             :   Yes (ver. 7.3.0)
    -- 
    -- Python:
    --   Interpreter       :   /usr/bin/python2.7 (ver. 2.7.15)
    --   Libraries         :   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.15rc1)
    --   NumPy             :   /home/creator/.local/lib/python2.7/site-packages/numpy/core/include (ver 1.16.2)
    -- 
    -- Matlab:
    --   Matlab            :   Yes (/usr/local/MATLAB/R2018a/bin/mex, /usr/local/MATLAB/R2018a/bin/mexext
    --   Octave            :   Yes (/usr/bin/mkoctfile)
    --   Build mex using   :   Matlab
    -- 
    -- Documentaion:
    --   Doxygen           :   /usr/bin/doxygen (1.8.13)
    --   config_file       :   /home/creator/Downloads/caffe-master/.Doxyfile
    -- 
    -- Install:
    --   Install path      :   /opt/caffe
    
    ```
    Matlab should be found.

* Make:
    ```bash
    # replace 8 with your cpu kernel count
    make all -j8
    sudo make install
    sudo chmod -R 777 /opt/caffe
    # remove origin caffe
    rm ~/.local/lib/python2.7/site-packages/caffe
    # make symbolic to python libs
    ln -s /opt/caffe/python/caffe ~/.local/lib/python2.7/site-packages/
    # copy matlab directory to install directory
    cp -r ../matlab/* /opt/caffe/matlab
    ```
* Modify `~/.bashrc`
    add 
    ```bash
    export PATH=/opt/caffe/bin:$PATH  # (optional)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/caffe/lib
    ``` 
    to the end of file `~/.bashrc`


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
if there is no error message, congratulations, you made it!

Install code hints file for Pycharm

See: https://github.com/Mannix1994/PythonResources

## 6. Test matcaffe

Open Matlab and write test.m
```matlab
PATH_TO_CAFFE_MATLAB='/opt/caffe/matlab';
addpath(genpath(PATH_TO_CAFFE_MATLAB));
caffe.set_mode_cpu();
```
Run test.m, there are some errors:
```txt
Invalid MEX-file '/opt/caffe/matlab/+caffe/private/caffe_.mexa64': /usr/lib/libgdal.so.20: symbol
TIFFReadRGBATileExt version LIBTIFF_4.0 not defined in file libtiff.so.5 with link time reference.

Error in caffe.set_mode_cpu (line 5)
caffe_('set_mode_cpu');

Error in test (line 3)
caffe.set_mode_cpu();
```
It looks like the lib LIBTIFF's problem. This is because Matlab 
will load the library of its self, we compile caffe and link
system libtiff. so we should preload system libtiff.

Firstly, find where libtiff is:
```bash
locate libtiff.so
```
output:
```txt
/snap/gimp/113/usr/lib/x86_64-linux-gnu/libtiff.so.5
...
/snap/vlc/770/usr/lib/x86_64-linux-gnu/libtiff.so.5.2.4
/usr/lib/x86_64-linux-gnu/libtiff.so
/usr/lib/x86_64-linux-gnu/libtiff.so.5
/usr/lib/x86_64-linux-gnu/libtiff.so.5.3.0
/usr/local/MATLAB/R2018a/bin/glnxa64/libtiff.so.5
/usr/local/MATLAB/R2018a/bin/glnxa64/libtiff.so.5.0.5
```
We should choose the system libtiff 
`/usr/lib/x86_64-linux-gnu/libtiff.so.5`not the one 
in `snap/`  or the one in `/usr/local/MATLAB/`.  
Open terminal and run:
```bash
# preload some libs
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
matlab  # run matlab from terminal
```
Then run test.m in matlab prompt, there will be no error. When you run 
matcaffe, if there are any problem about lib, you can solve it by the method
mentioned above.  
Some other libs need to be preload:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5:/usr/lib/x86_64-linux-gnu/libhdf5_cpp.so:/usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so:/usr/lib/x86_64-linux-gnu/libhdf5_serial.so:/usr/lib/x86_64-linux-gnu/libhdf5_serial_fortran.so:/usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so:/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
matlab # run matlab from terminal
```

Good luck.