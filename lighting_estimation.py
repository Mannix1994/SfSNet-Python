# coding=utf8

from SfSNet.sfsnet import SfSNet
from SfSNet.config import *
import cv2
import numpy as np

if __name__ == '__main__':
    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    image = cv2.imread('SfSNet/Images/0001_01.jpg')

    face, mask, normal, albedo, reconstruction, shading = sfsnet.forward(image)

    print np.max(face), np.min(face)
    print np.max(mask), np.min(mask)
    print np.max(normal), np.min(normal)
    print np.max(albedo), np.min(albedo)
    print np.max(reconstruction), np.min(reconstruction)
    print np.max(shading), np.min(shading)

    cv2.imshow('face', face)
    cv2.imshow('mask', mask)
    cv2.imshow('albedo', albedo)
    cv2.imshow('reconstruction', reconstruction)
    cv2.imshow('shading', shading)
    cv2.waitKey(0)
