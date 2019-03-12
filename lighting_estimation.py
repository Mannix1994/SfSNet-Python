# coding=utf8

import cv2
import numpy as np
import time

def draw_arrow(image, magnitude, angle, magnitude_threshold=1.0, length=10):
    # _image = image.copy()
    _image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    angle = angle / 180.0 * np.pi
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
    # h_kernel = np.array((-1, -1, 0, 1, 1)).reshape(1, 5)
    h_kernel = np.array((-1, -1, 0, 1, 1)).reshape(1, -1)
    # define vertical filter kernel
    v_kernel = h_kernel.T.copy()
    # kennel_x = np.array([[-1, 0, 1],
    #                      [-1, 0, 1],
    #                      [-1, 0, 1]])
    # kennel_y = kennel_x.T.copy()
    # filter horizontally
    h_conv = cv2.filter2D(gray, -1, kernel=h_kernel)
    # filter vertical
    v_conv = cv2.filter2D(gray, -1, kernel=v_kernel)
    # compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(h_conv, v_conv, angleInDegrees=True)
    if mask is not None:
        _mask = mask[:, :, 0]
        _mask = _mask / 255
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
    # please see light_estimation_分区.png
    right_down_1 = np.sum(np.int32((angle > 0) & (angle < 45)))
    right_down_2 = np.sum(np.int32((angle >= 45) & (angle < 90)))
    left_down_3 = np.sum(np.int32((angle >= 90) & (angle < 135)))
    left_down_4 = np.sum(np.int32((angle >= 135) & (angle < 180)))
    left_up_5 = np.sum(np.int32((angle >= 180) & (angle < 225)))
    left_up_6 = np.sum(np.int32((angle >= 225) & (angle < 270)))
    right_up_7 = np.sum(np.int32((angle >= 270) & (angle < 315)))
    right_up_8 = np.sum(np.int32((angle >= 315) & (angle < 360)))

    angle_in_range = [[1, right_down_1],
                      [2, right_down_2],
                      [3, left_down_3],
                      [4, left_down_4],
                      [5, left_up_5],
                      [6, left_up_6],
                      [7, right_up_7],
                      [8, right_up_8]]
    # angle_in_range = {'right_down_1': right_down_1,
    #                   'right_down_2': right_down_2,
    #                   'left_down_3': left_down_3,
    #                   'left_down_4': left_down_4,
    #                   'left_up_5': left_up_5,
    #                   'left_up_6': left_up_6,
    #                   'right_up_7': right_up_7,
    #                   'right_up_8': right_up_8,
    #                   }
    direction = _which_direction(angle_in_range)

    return direction, angle_in_range


def gray_level(shading, mask):
    """
    按照shading的像素平均值，把图像亮度分成七个等级。
    分别为：70以下\70-100\100-130\130-160\160-190\
    190-210\210-255
    :param shading:
    :param mask:
    :return:
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]/255
    pixel_count = np.sum(mask)

    shading_count = np.sum(shading)

    avg_pixel_val = shading_count / pixel_count
    print 'avg_pixel_val = ', avg_pixel_val
    level = 0
    if avg_pixel_val < 70:
        level = 0
    elif avg_pixel_val < 100:
        level = 1
    elif avg_pixel_val < 130:
        level = 2
    elif avg_pixel_val < 160:
        level = 4
    elif avg_pixel_val < 190:
        level = 5
    elif avg_pixel_val < 210:
        level = 6
    else:
        level = 7
    return avg_pixel_val, level


def _which_direction(angle_in_range):
    # please see light_estimation_方向.png
    directions = []
    _max = max(angle_in_range, key=lambda x: x[1])[1]
    _max = float(_max)
    _avg_angle_in_range = [(r, round(l / _max, 2)) for r, l in angle_in_range]
    s = sorted(_avg_angle_in_range, key=lambda x: x[1], reverse=True)

    print 's=', s
    # 前四个的占比都超过0.65
    if s[3][1] > 0.65:
        # 排序
        ss = sorted(s[0:4], key=lambda x: x[0])
        print 'ss=', ss
        # 连续的三种特殊情况
        zheyelianxu = [[1, 6, 7, 8], [1, 2, 7, 8], [1, 2, 3, 8]]
        xuhao = [sss[0] for sss in ss]
        if xuhao in zheyelianxu:
            if xuhao == [1, 6, 7, 8]:
                return 7
            elif xuhao == [1, 2, 7, 8]:
                return 8
            else:
                return 1
        else:
            # 连续，返回中间的值
            if xuhao[3] - xuhao[0] == 3:
                return (xuhao[1]+xuhao[2])/2.0
            else:
                # 如果不连续，那么认为是均匀光照
                return -1
    # 如果前两个的占比之差超过0.16，直接选第一个
    if (s[0][1] - s[1][1]) > 0.16:
        return s[0][0]
    # 如果第二个和第三个的差大于0.1，则取前两个的平均值
    elif (s[1][1] - s[2][1]) > 0.1:
        # 序号是1和8，是连续的，直接返回8
        if (s[1][0] == 8 and s[0][0] == 1) or (s[1][0] == 1 and s[0][0] == 8):
            return 8
        # 序号之差大于2，认为是均匀光照
        elif abs(s[1][0] - s[0][0]) > 2:
            return -1
        # 否则取平均值
        else:
            return (s[1][0] + s[0][0]) / 2.0
    # 第三个和第四个的差大于0.1，则认为前三个连续
    elif (s[2][1] - s[3][1]) > 0.1:
        # 排序
        ss = sorted(s[0:3], key=lambda x: x[0])
        print 'ss=', ss
        # 计算两两序号之差
        _1_0 = ss[1][0] - ss[0][0]
        _2_1 = ss[2][0] - ss[1][0]
        _2_0 = ss[2][0] - ss[0][0]
        # 序号是128和178是连续的
        if (np.array(ss)[:, 0] == np.array([1, 2, 8])).all():
            print '128--------------------128---------'
            return ss[0][0]
        elif (np.array(ss)[:, 0] == np.array([1, 7, 8])).all():
            print '178--------------------178---------'
            return ss[2][0]
        # 序号之差大于1，说明这三个序号不连续，认为是平均光照
        if _1_0 > 1 or _2_1 > 1:
            return -1
        # 三个序号连续，取中间的序号
        elif _1_0 == 1 and _2_1 == 1:
            return ss[1][0]
        # 没啥用
        else:
            return -1
    # 有四个连续，则直接判断为均匀光照
    else:
        return -1


if __name__ == '__main__':
    image = cv2.imread('shading.png', cv2.IMREAD_GRAYSCALE)
    print which_direction(image, None, 1, True)
