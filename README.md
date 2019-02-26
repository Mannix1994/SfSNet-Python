# Introduction
This project partially implement SfSNet project. 
I will implement all codes in SfSNet project in the future,
if I have enough time.

Currently implemented:
* Implement test_SfSNet.m as _test_SfSNet.py
* Partially implement functions/*m in functions.py
* move some constant variables to config.py 

# Dependencies
* Python 2.7
* Caffe with pycaffe module
* Python libs in requirements.txt

# Run _test_SfSNet.py

You should modify the variable CAFFE_ROOT in config.py 
to your own caffe install directory. Then, install python
dependencies using bash command:
```bash
pip install -r requirements.txt
```
then put your test images in 'Images', and 
```bash
python _test_SfSNet.py
```
and input 0 when the program tell you to input. the results
will be saved in directory 'result' 
