# coding=utf8
import numpy as np
import sys
import shutil
import cv2
import glob
import csv
import traceback
from config import *
from lighting_estimation import which_direction, gray_level, Statistic, gray_level_keys
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


direction_keys = ['left', 'right', 'frontal']


def found_processed_ids(filename, statistic_ids, index=0, records=[]):
    with open(filename, 'r') as f:
        next(f)
        r = csv.reader(f)
        for i in r:
            if i[index] not in statistic_ids:
                records.append(i[index:])
                statistic_ids.add(i[index])


def save_csv(filename, records, keys):
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerow(['id', ] + keys)
        for r in records:
            w.writerow(r)


def save(file, keys):
    statistic_ids = set([])
    record_levels = []
    found_processed_ids('data analysis/1/%s.csv' % file, statistic_ids, 0, record_levels)
    found_processed_ids('data analysis/2/%s.csv' % file, statistic_ids, 0, record_levels)
    found_processed_ids('data analysis/3/%s.csv' % file, statistic_ids, 0, record_levels)
    found_processed_ids('data analysis/5/%s.csv' % file, statistic_ids, 0, record_levels)
    found_processed_ids('data analysis/6/%s.csv' % file, statistic_ids, 0, record_levels)
    found_processed_ids('data analysis/7~n/%s.csv' % file, statistic_ids, 0, record_levels)
    print len(statistic_ids), len(record_levels)
    save_csv('total_%s.csv' % file, record_levels, keys)


if __name__ == '__main__':
    save('level', gray_level_keys)
    save('direction', direction_keys)
    dir_level_keys = ['%s_%s' % (_d, _l) for _d in direction_keys for _l in gray_level_keys]
    save('dir_level', dir_level_keys)


def ijb_a(show=False):
    statistic_ids = set([])
    found_processed_ids('data analysis/1/level.csv', statistic_ids, 0, )
    found_processed_ids('data analysis/2/level.csv', statistic_ids, 0, )
    found_processed_ids('data analysis/3/level.csv', statistic_ids, 0, )
    found_processed_ids('data analysis/5/level.csv', statistic_ids, 0, )
    found_processed_ids('data analysis/6/level.csv', statistic_ids, 0, )
    found_processed_ids('data analysis/7~n/level.csv', statistic_ids, 0, )
    print len(statistic_ids)
    exit()
    # 默认存储目录
    base_dir = os.path.join(PROJECT_DIR, 'result')
    # SfSNet实例，自己把sfenet包装了一遍
    sfsnet = SfSNet(MODEL, WEIGHTS, GPU_ID, LANDMARK_PATH)

    # 统计光照方向
    direction_sta = Statistic('direction.csv', True, *direction_keys)
    # 统计shading的分布
    level_keys = gray_level_keys
    level_sta = Statistic('level.csv', True, *level_keys)
    # 统计方向与光照的组合
    dir_level_keys = ['%s_%s' % (_d, _l) for _d in direction_keys for _l in level_keys]
    dir_level_sta = Statistic('dir_level.csv', True, *dir_level_keys)

    try:
        for _index in range(1, 11, 1):
            # 列表文件
            list_file = os.path.join(IJB_A_11, 'split'+str(_index), 'train_%d.csv' % _index)
            # 包括人物id，文件名，以及人脸正方形(左上角定点，人脸的的宽和高)
            people_records = []

            if show:
                cv2.namedWindow('face', cv2.WINDOW_NORMAL)
                cv2.namedWindow('shading', cv2.WINDOW_NORMAL)
            gray_val = [256, ]

            with open(list_file, mode='r') as f:
                # 跳过IJB-A列表文件的标题行
                next(f)
                # 定义CSV读取器
                reader = csv.reader(f)
                # 读取所有的人，并把需要的信息提取出来
                for line in reader:
                    people_records.append([line[1], line[2],
                                           (int(float(line[6])), int(float(line[7])),
                                            int(float(line[8])), int(float(line[9])))])
                for record in people_records:
                    print '*' * 120
                    print record  # ['417', 'frame/28819_00060.png', (116, 16, 116, 138)]
                    if record[0] in statistic_ids:
                        continue
                    # 从record里读取人脸
                    image, rect = crop_face_from_image(record, show=False)
                    # 对齐和裁剪人脸
                    mask, aligned_image = sfsnet.process_image(image, show=False)
                    if mask is not None:
                        # 对齐成功，mask不为空
                        face, mask, _, _, _, shading = sfsnet.forward(aligned_image, mask)
                    else:
                        # 对齐失败
                        shape = image.shape[0:2]
                        # 调整大小
                        resize_image = cv2.resize(image, (M, M))
                        # 使用未对齐的人脸计算shading
                        face, mask, _, _, _, shading = sfsnet.forward(resize_image, None)
                        # 调整回原来的大小
                        shading = cv2.resize(shading, shape)
                        # 裁剪出人脸
                        shading = shading[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
                        print('shading.shape', shading.shape)
                    # 计算人脸方向和统计梯度的角度
                    direction, angle_count = which_direction(shading, mask, magnitude_threshold=10)
                    angle_count = sorted(angle_count, key=lambda x: x[1], reverse=True)
                    print(direction, angle_count)

                    # 计算shading属于那个亮度等级level
                    avg_pixel_val, level = gray_level(shading, mask)
                    print('avg_pixel_val =', avg_pixel_val, 'level =', level)
                    gray_val.append(avg_pixel_val)

                    # 写入统计数据
                    direction_sta.add(record[0], conclude_direction(direction))
                    level_sta.add(record[0], level)
                    dir_level_sta.add(record[0], '%s_%s' % (conclude_direction(direction), level))

                    # 默认存储路径
                    id_dir = os.path.join(base_dir, record[0], str(level))
                    # 不存在则新建图像
                    if not os.path.exists(id_dir):
                        os.makedirs(id_dir)
                    # 存储图像
                    cv2.imwrite(os.path.join(id_dir, record[1].split('/')[-1]), face)
                    if show:
                        cv2.imshow('face', face)
                        cv2.imshow('shading', shading)
                        if cv2.waitKey(50) == 27:
                            print 'Exiting...'
                            exit()
                print np.max(gray_val), np.min(gray_val)
    except:
        traceback.print_exc()
    finally:
        # 保存统计信息
        direction_sta.save()
        level_sta.save()
        dir_level_sta.save()


def conclude_direction(direction):
    if 3 < direction < 5.5:
        return direction_keys[0]
    elif 6.5 < direction <= 8 or 0 < direction <= 1:
        return direction_keys[1]
    else:
        return direction_keys[2]


if __name__ == '__main__':
    ijb_a(False)
