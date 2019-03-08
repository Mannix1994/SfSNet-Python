# coding=utf8

from __future__ import absolute_import, division, print_function

import dlib
import cv2
import sys
import numpy as np
from PIL import Image
from functions import create_mask_fiducial
from mtcnn_pytorch.mtcnn import MTCNN
import logging


class MaskGenerator:
    def __init__(self, landmarks_path='shape_predictor_68_face_landmarks.dat'):
        """
        :param landmarks_path: the path of pretrained key points weight,
        it could be download from:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(landmarks_path)
        self._d = MTCNN()

    def get_mask(self, image, crop_size=(128, 128), scale=3.5,
                 show_rectangle=False, show_landmarks=False):
        """
        :param show_landmarks:
        :param show_rectangle:
        :param scale:
        :param crop_size:
        :type image: np.ndarray
        :param image: BGR face image
        :return: a face image with mask
        https://blog.csdn.net/qq_39438636/article/details/79304130
        """
        try:
            boxes, face = self._d.align(image, crop_size, scale)
            face_rects = []
            # 转换为rectangle对象
            for b in boxes:
                face_rects.append(dlib.rectangle(*b))
            b_im = face.copy()
            # 显示矩形
            if show_rectangle:
                cv2.rectangle(b_im, (face_rects[0].left(), face_rects[0].top()),
                              (face_rects[0].right(), face_rects[0].bottom()), (255, 255, 0))
                cv2.imshow("b_im", b_im)
                cv2.waitKey(50)
            # 计算landmarks
            for i in range(len(face_rects)):
                landmarks = np.array([[p.x, p.y] for p in self._predictor(face, face_rects[i]).parts()])
                mask = create_mask_fiducial(landmarks.T, face)

                if show_landmarks:
                    self._view_landmarks(face, landmarks)
                return mask, face
        except BaseException as e:
            sys.stderr.write("%s: failed to align\n" % __file__)
            logging.exception(e)
            return np.ones(image.shape) * 255, image

    def get_masked_face(self, image):
        mask, face = self.get_mask(image)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(50)
        return mask & face

    def _view_landmarks(self, image, landmarks):
        im = image.copy()
        for landmark in landmarks:
            cv2.circle(im, tuple(landmark), 1, (0, 255, 0))
        cv2.imshow('landmarks', im)
        cv2.waitKey(50)


if __name__ == '__main__':
    image = cv2.imread('Images/0003_01.jpg')
    mask_gen = MaskGenerator()
    mah= mask_gen.get_masked_face(image)
    cv2.imshow('123', mah)
    cv2.waitKey(0)
