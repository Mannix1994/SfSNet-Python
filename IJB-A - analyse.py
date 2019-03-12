# coding=utf8
import numpy as np
import sys
import shutil
import cv2
import glob
import csv
from config import *
from lighting_estimation import which_direction, gray_level, Statistic
from SfSNet.sfsnet import SfSNet
from ijb_a import crop_face_from_image


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

        face, mask, shape, albedo, reconstruction, shading = sfsnet.forward_with_align(image, show=False)

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
    base_dir = os.path.join(PROJECT_DIR, 'result')

    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    list_file = os.path.join(IJB_A_11, 'split1', 'train_1.csv')
    people_records = []

    if show:
        cv2.namedWindow('face', cv2.WINDOW_NORMAL)
        cv2.namedWindow('shading', cv2.WINDOW_NORMAL)
    gray_val = []
    # shabi

    direction_keys = ['left', 'right', 'direct']
    level_keys = [0, 1, 2, 3, 4, 5]
    direction_sta = Statistic('direction.csv', *direction_keys)
    level_sta = Statistic('level.csv', *level_keys)
    dir_level_keys = ['%s_%d' % (_d, _l) for _d in direction_keys for _l in level_keys]
    dir_level_sta = Statistic('dir_level.csv', *dir_level_keys)

    with open(list_file, mode='r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            people_records.append([line[1], line[2],
                                   (int(float(line[6])), int(float(line[7])),
                                    int(float(line[8])), int(float(line[9])))])
        for record in people_records[0:10]:
            print '*' * 120
            print record
            image, rect = crop_face_from_image(record, show=False)
            mask, aligned_image = sfsnet.process_image(image, show=False)
            if mask is not None:
                face, mask, _, _, _, shading = sfsnet.forward(aligned_image, mask)
            else:
                shape = image.shape[0:2]
                resize_image = cv2.resize(image, (M, M))
                face, mask, _, _, _, shading = sfsnet.forward(resize_image, None)
                shading = cv2.resize(shading, shape)
                shading = shading[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

            direction, angle_count = which_direction(shading, mask, magnitude_threshold=10)
            angle_count = sorted(angle_count, key=lambda x: x[1], reverse=True)
            print direction, angle_count

            avg_pixel_val, level = gray_level(shading, mask)
            print 'avg_pixel_val =', avg_pixel_val, 'level =', level
            gray_val.append(avg_pixel_val)

            # 写入统计数据
            direction_sta.add(record[0], conclude_direction(direction))
            level_sta.add(record[0], level)
            dir_level_sta.add(record[0], '%s_%d' % (conclude_direction(direction), level))

            id_dir = os.path.join(base_dir, record[0], conclude_direction(direction))
            if not os.path.exists(id_dir):
                os.makedirs(id_dir)
            cv2.imwrite(os.path.join(id_dir, record[1].split('/')[-1]), face)
            if show:
                cv2.imshow('face', face)
                cv2.imshow('shading', shading)
                if cv2.waitKey(1) == 27:
                    print 'Exiting...'
                    exit()
        print np.max(gray_val), np.min(gray_val)
        direction_sta.save()
        level_sta.save()


def conclude_direction(direction):
    if 3 < direction < 5.5:
        return 'left'
    elif 6.5 < direction <= 8 or 0 < direction <= 1:
        return 'right'
    else:
        return 'direct'


if __name__ == '__main__':
    ijb_a(True)
