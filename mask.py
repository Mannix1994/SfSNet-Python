# coding=utf8

from __future__ import absolute_import, division, print_function

import dlib
import cv2
import sys
import numpy as np
from PIL import Image
from functions import create_mask_fiducial


class MaskGenerator:
    def __init__(self, landmarks_path='shape_predictor_68_face_landmarks.dat'):
        """
        :param landmarks_path: the path of pretrained key points weight,
        it could be download from:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(landmarks_path)

    def get_mask(self, image):
        """
        :type image: np.ndarray
        :param image: BGR face image
        :return: a face image with mask
        https://blog.csdn.net/qq_39438636/article/details/79304130
        """
        # convert to gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测人脸矩形
        face_rects = self._detector(gray_image, 0)
        for i in range(len(face_rects)):
            landmarks = np.array([[p.x, p.y] for p in self._predictor(image, face_rects[i]).parts()])
            mask = create_mask_fiducial(landmarks.T, image)
            return mask
        else:
            sys.stderr.write("%s: Can't detect face in image\n" % __file__)
            return np.ones(image.shape) * 255

    def get_masked_face(self, image):
        mask = self.get_mask(image)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(50)
        return mask & image


if __name__ == '__main__':
    image = cv2.imread('Images/0003_01.jpg')
    mask_gen = MaskGenerator()
    mah= mask_gen.get_masked_face(image)
    cv2.imshow('123', mah)
    cv2.waitKey(0)
