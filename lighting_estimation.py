# coding=utf8

import cv2
import numpy as np


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


def which_direction(image, mask, magnitude_threshold=1.0, show_arrow=False):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(image)
    # define horizontal filter kernel
    h_kernel = np.array((-1, -1, 0, 1, 1)).reshape(1, 5)
    # define vertical filter kernel
    v_kernel = np.array((-1, -1, 0, 1, 1)).reshape(5, 1)
    # filter horizontally
    h_conv = cv2.filter2D(gray, -1, kernel=h_kernel)
    # filter vertical(rotate)
    v_conv = cv2.filter2D(gray, -1, kernel=v_kernel)
    # compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(h_conv, v_conv, angleInDegrees=True)
    if mask is not None:
        _mask = mask[:, :, 0]
        _mask = _mask/255
        # remove the un-masked area
        magnitude *= _mask
        angle *= _mask
    # draw some arrow
    if show_arrow:
        im = draw_arrow(image, magnitude, angle, magnitude_threshold)
        cv2.namedWindow('arrow', cv2.WINDOW_NORMAL)
        cv2.imshow('arrow', im)
        cv2.waitKey(50)
    # set angle[i,j]=0 if magnitude[i, j] < magnitude_threshold
    angle = angle * np.int32(magnitude > magnitude_threshold)
    # count the angle's direction
    # please see light_estimation_坐标系.png
    right_down_1 = np.sum(np.int32((angle > 0) & (angle < 45)))
    right_down_2 = np.sum(np.int32((angle >= 45) & (angle < 90)))
    left_down_3 = np.sum(np.int32((angle >= 90) & (angle < 135)))
    left_down_4 = np.sum(np.int32((angle >= 135) & (angle < 180)))
    left_up_5 = np.sum(np.int32((angle >= 180) & (angle < 225)))
    left_up_6 = np.sum(np.int32((angle >= 225) & (angle < 270)))
    right_up_7 = np.sum(np.int32((angle >= 270) & (angle < 315)))
    right_up_8 = np.sum(np.int32((angle >= 315) & (angle < 360)))

    return [('1', right_down_1),
            ('2', right_down_2),
            ('3', left_down_3),
            ('4', left_down_4),
            ('5', left_up_5),
            ('6', left_up_6),
            ('7', right_up_7),
            ('8', right_up_8),
            ]
    # return {'right_down_1': right_down_0,
    #         'right_down_2': right_down_1,
    #         'left_down_3': left_down_2,
    #         'left_down_4': left_down_3,
    #         'left_up_5': left_up_4,
    #         'left_up_6': left_up_5,
    #         'right_up_7': right_up_6,
    #         'right_up_8': right_up_7,
    #         }


if __name__ == '__main__':
    image = cv2.imread('shading.png', cv2.IMREAD_GRAYSCALE)
    print which_direction(image, None, 1, True)
