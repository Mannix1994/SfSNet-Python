# coding=utf-8
import os

CAFFE_ROOT = "/home/creator/Apps/caffe"

# Project directory
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

MODEL = os.path.join(PROJECT_DIR, 'model/SfSNet_deploy.prototxt')

WEIGHTS = os.path.join(PROJECT_DIR, 'weights/SfSNet.caffemodel.h5')

GPU_ID = 0

M = 128  # size of input for SfSNet

if __name__ == '__main__':
    print PROJECT_DIR
    print MODEL
    print WEIGHTS
