# coding=utf-8
import os

# Caffe install directory
CAFFE_ROOT = "/home/creator/Apps/caffe"

# Project directory
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

# model's path
MODEL = os.path.join(MODULE_DIR, 'model/SfSNet_deploy.prototxt')

# weights's path
WEIGHTS = os.path.join(MODULE_DIR, 'weights/SfSNet.caffemodel.h5')

# gpu's id
GPU_ID = 0

# image's size, DO NOT CHANGE!
M = 128  # size of input for SfSNet

# landmarks's path
LANDMARK_PATH = 'shape_predictor_68_face_landmarks.dat'

if __name__ == '__main__':
    print MODULE_DIR
    print MODEL
    print WEIGHTS
