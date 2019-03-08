# coding=utf8

from SfSNet import SfSNet
from config import *
import cv2

if __name__ == '__main__':
    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    image = cv2.imread('SfSNet/Images/0001_01.jpg')

    face, shape, albedo, reconstruction, shading = sfsnet.forward(image)
