# coding=utf8
import numpy as np
import sys
import shutil
import cv2
from config import *
import glob
from lighting_estimation import which_direction, gray_level
from SfSNet.sfsnet import SfSNet

if __name__ == '__main__':

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

        face, mask, shape, albedo, reconstruction, shading = sfsnet.forward(image, show=False)

        # print face.shape, shape.shape, albedo.shape, reconstruction.shape, shading.shape
        if mask is not None:
            print '*' * 120
            direction, result = which_direction(shading, mask, show_arrow=False)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            print direction, result

            gray_val.append(gray_level(shading, mask))

            cv2.imwrite(os.path.join('result', str(int(direction)), im.split('/')[-1]), shading)

            cv2.imshow('face', face)
            cv2.imshow('shading', shading)
            key = cv2.waitKey(1)
            if key == 27:
                print 'Exiting...'
                exit()
    print np.max(gray_val), np.min(gray_val)

IJB_A_ROOT = '/home/creator/E/wangmz/Ubuntu/IJB/IJB-A'
IJB_A_11 = os.path.join(IJB_A_ROOT, 'IJB-A_11_sets')
IJB_A_IMAGE_ROOT = os.path.join(IJB_A_ROOT, 'images')


def crop_face_from_image(record, scale=0.01):
    """
    crop image.
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
    face_x = int(record[2][0]*(1-0.12))
    face_y = int(record[2][1]*(1-0.12))
    face_width = int(record[2][2]*(1+0.9))
    face_height = int(record[2][3]*(1+0.6))
    cv2.rectangle(image, (face_x, face_y), (face_x+face_width, face_y+face_height),
                  (0, 255, 0))
    cv2.imshow('image', image)
    cv2.waitKey(1)
    # 截取人脸
    cropped_image = image[face_y:face_y+face_height, face_x:face_x+face_width]
    if cropped_image.shape[0] > 500:
        # 等比例缩小
        ratio = cropped_image.shape[0]/500.0
        cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1]/ratio),
                                                   int(cropped_image.shape[0]/ratio)))
    return cropped_image


if __name__ == '__main__':
    list_file = os.path.join(IJB_A_11, 'split1', 'train_1.csv')
    people_records = []
    with open(list_file, mode='r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            people_records.append([line[1], line[2],
                                   (int(float(line[6])), int(float(line[7])),
                                 int(float(line[8])), int(float(line[9])))])
            print(people_records[-1])
            image = crop_face_from_image(people_records[-1])
            cv2.imshow('ss', image)
            if cv2.waitKey(0) == 27:
                exit()