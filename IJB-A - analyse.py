# coding=utf8
import numpy as np
import sys
import shutil
import cv2
import glob
import csv
from config import *
from lighting_estimation import which_direction, gray_level
from SfSNet.sfsnet import SfSNet


def vggface():

    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, 'shape_predictor_68_face_landmarks.dat')

    images = glob.glob("/home/creator/E/wangmz/Ubuntu/VGGFace2-train/n000230/*.*")
    print images
    for i in range(-1, 9, 1):
        shutil.rmtree(os.path.join('result', str(i)), ignore_errors=True)
        os.mkdir(os.path.join('result', str(i)))
    gray_val = []

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.namedWindow('shading', cv2.WINDOW_NORMAL)
    for im in images:
        image = cv2.imread(im)
        if image is None:
            sys.stderr.write("Empty image: " + im)
            continue

        face, mask, shape, albedo, reconstruction, shading = sfsnet.forward(image, show=True)

        # print face.shape, shape.shape, albedo.shape, reconstruction.shape, shading.shape
        if mask is not None:
            print '*' * 120
            direction, result = which_direction(shading, mask, magnitude_threshold=3.0, show_arrow=False)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            print direction, result

            gray_val.append(gray_level(shading, mask)[0])

            cv2.imwrite(os.path.join('result', str(int(direction)), im.split('/')[-1]), shading)

            cv2.imshow('face', face)
            cv2.imshow('shading', shading)
            key = cv2.waitKey(0)
            if key == 27:
                print 'Exiting...'
                exit()
    print np.max(gray_val), np.min(gray_val)


def ijb_a(show=False):

    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    list_file = os.path.join(IJB_A_11, 'split1', 'train_1.csv')
    people_records = []

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.namedWindow('shading', cv2.WINDOW_NORMAL)
    gray_val = []
    with open(list_file, mode='r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            people_records.append([line[1], line[2],
                                   (int(float(line[6])), int(float(line[7])),
                                    int(float(line[8])), int(float(line[9])))])
        for record in people_records:
            print record
            image = crop_face_from_image(record, show=False)
            if show:
                cv2.imshow('crop_face', image)
                if cv2.waitKey(1) == 27:
                    exit()
            face, mask, shape, albedo, reconstruction, shading = sfsnet.forward(image, show=False)

            if mask is not None:
                print '*' * 120
                direction, result = which_direction(shading, mask, show_arrow=False)
                result = sorted(result, key=lambda x: x[1], reverse=True)
                print direction, result

                gray_val.append(gray_level(shading, mask)[0])

                # cv2.imwrite(os.path.join('result', str(int(direction)), im.split('/')[-1]), shading)

                if show:
                    cv2.imshow('face', face)
                    cv2.imshow('shading', shading)
                    if cv2.waitKey(0) == 27:
                        print 'Exiting...'
                        exit()
        print np.max(gray_val), np.min(gray_val)


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
    if cropped_image.shape[0] > 500:
        # 等比例缩小
        ratio = cropped_image.shape[0] / 500.0
        cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1] / ratio),
                                                   int(cropped_image.shape[0] / ratio)))
    return cropped_image


if __name__ == '__main__':
    vggface()
