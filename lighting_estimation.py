# coding=utf8

from SfSNet.sfsnet import SfSNet
from SfSNet.config import *
import cv2
import numpy as np

if __name__ == '1':
    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    image = cv2.imread('SfSNet/Images/0001_01.jpg')

    face, mask, normal, albedo, reconstruction, gray = sfsnet.forward(image)

    # cv2.imshow('face', face)
    # cv2.imshow('mask', mask)
    # cv2.imshow('albedo', albedo)
    # cv2.imshow('reconstruction', reconstruction)
    cv2.imshow('shading', gray)
    cv2.imwrite('shading.png', gray)
    cv2.waitKey(0)


def draw_arrow(image, magnitude, angle, magnitude_threshold=1.0, length=10):
    # _image = image.copy()
    _image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    angle = angle/180.0*np.pi
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            magni = magnitude[i, j]
            ang = angle[i, j]
            if magni < magnitude_threshold:
                continue
            diff_i = int(np.round(np.sin(ang) * length))
            diff_j = int(np.round(np.cos(ang) * length))
            cv2.line(_image, (j, i), (j + diff_j, i + diff_i), (0, 255, 0))
            p_i = np.max((0, i + diff_i))
            p_i = np.min((_image.shape[0] - 1, p_i))
            p_j = np.max((0, j + diff_j))
            p_j = np.min((_image.shape[1] - 1, p_j))
            _image[p_i, p_j] = (0, 0, 255)
    return _image


def which_direction(image, magnitude_threshold=1.0, show_arrow=False):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(image)
    # define horizontal filter kernel
    h_kernel = (-1, 0, 1)
    # define vertical filter kernel
    v_kernel = (-1, 0, 1)
    # filter horizontally
    h_conv = cv2.filter2D(gray, -1, kernel=h_kernel)
    # filter vertical(rotate)
    v_conv = cv2.filter2D(cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE), -1, kernel=v_kernel)
    v_conv = cv2.rotate(v_conv, cv2.ROTATE_90_CLOCKWISE)
    # compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(h_conv, v_conv, angleInDegrees=True)
    # draw some arrow
    if show_arrow:
        im = draw_arrow(image, magnitude, angle, magnitude_threshold)
        cv2.namedWindow('arrow', cv2.WINDOW_NORMAL)
        cv2.imshow('arrow', im)
        cv2.waitKey(0)
    # set angle[i,j]=0 if magnitude[i, j] < magnitude_threshold
    angle = angle * np.int32(magnitude > magnitude_threshold)
    # count the angle's direction
    right_down = np.sum(np.int32((angle > 0) & (angle < 90)))
    left_down = np.sum(np.int32((angle > 90) & (angle < 180)))
    left_up = np.sum(np.int32((angle > 180) & (angle < 270)))
    right_up = np.sum(np.int32((angle > 270) & (angle < 360)))
    return {'right_down': right_down,
            'left_down': left_down,
            'left_up': left_up,
            'right_up': right_up}


if __name__ == '__main__':
    image = cv2.imread('shading.png', cv2.IMREAD_GRAYSCALE)
    print which_direction(image, 1, True)
