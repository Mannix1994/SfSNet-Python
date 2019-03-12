# coding=utf8
import sys
import cv2
from config import *
import csv

def crop_face_from_image(record, scale=0.8, show=False):
    """
    crop image.
    :param scale: the scale want crop
    :param record: a record is like: ['966', 'frame/29456_00375.png', (129, 43, 102, 140)]
    '966': image's id; 'frame/29456_00375.png': image's path; 129: FACE_X, 43: FACE_Y,
    102: FACE_WIDTH, 140: FACE_HEIGHT.
    :return: cropped image
    """
    image_path = os.path.join(IJB_A_IMAGE_ROOT, str(record[1]).replace('frame', 'frames'))
    image = cv2.imread(image_path)
    if image is None:
        sys.stderr.write('Can\'t read image: %s \n' % image_path)
        exit()
    face_x = int(record[2][0])
    face_y = int(record[2][1])
    face_width = int(record[2][2])
    face_height = int(record[2][3])
    # 计算截取的区域中心点
    center_x = int(face_x + face_width/2)
    center_y = int(face_y + face_height/2)
    # 以中心以及高度和宽度的一定比例，增加裁剪的区域
    left_top_x = int(center_x - face_width * scale)
    left_top_y = int(center_y - face_height * scale)
    right_down_x = int(center_x + face_width * scale)
    right_down_y = int(center_y + face_height * scale)
    # 确保范围合法
    left_top_x = max(0, left_top_x)
    left_top_y = max(0, left_top_y)
    right_down_x = min(image.shape[1], right_down_x)
    right_down_y = min(image.shape[0], right_down_y)

    if show:
        im = image.copy()
        cv2.circle(im, (face_x, face_y), 10, (0, 0, 255))
        cv2.circle(im, (center_x, center_y), 10, (0, 255, 0))
        cv2.rectangle(im, (face_x, face_y), (face_x + face_width, face_y + face_height),
                      (0, 255, 0))
        cv2.rectangle(im, (left_top_x, left_top_y), (right_down_x, right_down_y),
                      (255, 0, 0))
        cv2.imshow('src image', im)
        cv2.waitKey(1)

    # 截取人脸
    cropped_image = image[left_top_y:right_down_y, left_top_x:right_down_x]

    # 计算人脸的在cropped_image里左上角和右下角的坐标
    new_left_top_x = face_x - left_top_x
    new_left_top_y = face_y - left_top_y
    new_right_down_x = (face_x + face_width) - left_top_x
    new_right_down_y = (face_y + face_height) - left_top_y
    # 确保范围合法
    new_left_top_x = max(0, new_left_top_x)
    new_left_top_y = max(0, new_left_top_y)
    new_right_down_x = min(cropped_image.shape[1], new_right_down_x)
    new_right_down_y = min(cropped_image.shape[0], new_right_down_y)

    return cropped_image, ((new_left_top_x, new_left_top_y), (new_right_down_x, new_right_down_y))

