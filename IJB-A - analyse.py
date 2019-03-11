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

    images = glob.glob("/home/creator/E/wangmz/Ubuntu/VGGFace2-train/n000219/*.*")
    print images
    for i in range(-1, 9, 1):
        shutil.rmtree(os.path.join('result', str(i)), ignore_errors=True)
        os.mkdir(os.path.join('result', str(i)))
    gray_val = []
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

            cv2.namedWindow('face', cv2.WINDOW_NORMAL)
            cv2.namedWindow('shading', cv2.WINDOW_NORMAL)
            cv2.imshow('face', face)
            cv2.imshow('shading', shading)
            key = cv2.waitKey(1)
            if key == 27:
                print 'Exiting...'
                exit()
    print np.max(gray_val), np.min(gray_val)