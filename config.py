# coding=utf-8
import os

# Caffe install directory
CAFFE_ROOT = "/home/creator/Apps/caffe"

# Project directory
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

# model's path
MODEL = os.path.join(PROJECT_DIR, 'SfSNet/model/SfSNet_deploy.prototxt')

# weights's path
WEIGHTS = os.path.join(PROJECT_DIR, 'SfSNet/weights/SfSNet.caffemodel.h5')

# gpu's id
GPU_ID = 0

# image's size, DO NOT CHANGE!
M = 128  # size of input for SfSNet

# landmarks's path
LANDMARK_PATH = os.path.join(PROJECT_DIR, 'shape_predictor_68_face_landmarks.dat')

# IJB-A
IJB_A_ROOT = '/home/creator/E/wangmz/Ubuntu/IJB/IJB-A'
IJB_A_11 = os.path.join(IJB_A_ROOT, 'IJB-A_11_sets')
IJB_A_IMAGE_ROOT = os.path.join(IJB_A_ROOT, 'images')

if __name__ == '__main__':
    print PROJECT_DIR
    print MODEL
    print WEIGHTS
