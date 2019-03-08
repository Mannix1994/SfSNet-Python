# coding=utf8

import numpy as np
import sys
import os
import shutil
import cv2
from matplotlib import pyplot as plt

from config import *
from functions import create_shading_recon

# the two lines add pycaffe support
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe
from mask import MaskGenerator


def _test():
    # set gpu mode, if you don't have gpu, use caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    # caffe.set_mode_cpu()

    # load model and weights
    net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)

    # choose dataset
    # dat_idx = input('Please enter 1 for images with masks and 0 for images without mask: ')
    dat_idx = 0
    if dat_idx:
        # Images and masks are provided
        list_im = sorted(os.listdir('Images_mask/'))
        list_im = [im for im in list_im if im.endswith('_face.png')]
        dat_idx = 1
    elif dat_idx == 0:
        # No mask provided (Need to use your own mask).
        list_im = sorted(os.listdir('Images/'))
        list_im = [im for im in list_im if im.endswith('.png') or im.endswith('.jpg')]
        dat_idx = 0  # Uncomment to test with this mode
    else:
        sys.stderr.write('Wrong Option!')
        list_im = None
        exit(-1)

    print list_im, dat_idx

    # define a mask generator
    mg = MaskGenerator(LANDMARK_PATH)

    l_image = []
    l_normal = []
    l_albedo = []
    l_recon = []
    l_shading = []
    # process every image
    for im_name in list_im:
        print 'Processing ' + im_name
        # read image
        if dat_idx == 1:
            # read face image as BGR format
            o_im = cv2.imread(os.path.join(PROJECT_DIR, 'Images_mask', im_name))
            im = o_im.copy()
            # resize image
            im = cv2.resize(im, (M, M))
            # get mask image's name
            mask_name = im_name.replace('face', 'mask')
            # read mask image as BGR format
            Mask = cv2.imread(os.path.join(PROJECT_DIR, 'Images_mask', mask_name))
            Mask = np.float32(Mask) / 255.0
            mask = cv2.resize(Mask, (M, M))
        else:
            o_im = cv2.imread(os.path.join(PROJECT_DIR, 'Images', im_name))
            im = o_im.copy()
            mask, im = mg.align(im, crop_size=(M, M))
            cv2.imshow("mask", mask)
            cv2.imshow("image", im)
            cv2.waitKey(50)

        # prepare image
        # im=reshape(im,[size(im)]);
        im = np.float32(im)/255.0  # im=single(im)/255;
        # im = np.transpose(im, [1, 0, 2])  # m_data = permute(im_data, [2, 1, 3]); switch width and height

        # -----------add by wang-------------
        im = np.transpose(im, [2, 0, 1])  # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.expand_dims(im, 0)
        # print 'im shape', im.shape
        # -----------end---------------------

        # set image data, pass images
        net.blobs['data'].data[...] = im

        # forward
        out_im = net.forward()
        n_out = out_im['Nconv0']  # normal, n_out=out_im{2};
        al_out = out_im['Aconv0']  # albedo, al_out=out_im{1};
        light_out = out_im['fc_light']  # shading, light_out=out_im{3};

        # -----------add by wang-------------
        # from [1, 3, 128, 128] to [128, 128, 3]
        n_out = np.squeeze(n_out, 0)
        n_out = np.transpose(n_out, [2, 1, 0])
        # from [1, 3, 128, 128] to [128, 128, 3]
        al_out = np.squeeze(al_out, 0)
        al_out = np.transpose(al_out, [2, 1, 0])
        # from [1, 27] to [27, 1]
        light_out = np.transpose(light_out, [1, 0])
        # print n_out.shape, al_out.shape, light_out.shape
        # -----------end---------------------


        """
        light_out is a 27 dimensional vector. 9 dimension for each channel of
        RGB. For every 9 dimensional, 1st dimension is ambient illumination
        (0th order), next 3 dimension is directional (1st order), next 5
        dimension is 2nd order approximation. You can simply use 27
        dimensional feature vector as lighting representation.
        """

        # transform
        n_out2 = n_out[:, :, (2, 1, 0)]
        # print 'n_out2 shape', n_out2.shape
        n_out2 = cv2.rotate(n_out2, cv2.ROTATE_90_CLOCKWISE)  # imrotate(n_out2,-90)
        n_out2 = np.fliplr(n_out2)
        n_out2 = 2*n_out2-1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2**2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
        nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2/np.repeat(nr, 3, axis=2)
        # print 'nr shape', nr.shape

        al_out2 = cv2.rotate(al_out, cv2.ROTATE_90_CLOCKWISE)
        al_out2 = al_out2[:, :, (2, 1, 0)]
        al_out2 = np.fliplr(al_out2)

        # Note: n_out2, al_out2, light_out is the actual output
        Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)

        diff = (mask/255)

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        n_out2 = n_out2 * diff
        al_out2 = al_out2 * diff
        Ishd = Ishd * diff
        Irec = Irec * diff

        Ishd = np.float32(Ishd)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_GRAY2RGB)

        l_image.append(o_im[:, :, [2, 1, 0]])
        l_albedo.append(al_out2)
        l_normal.append(n_out2)
        l_recon.append(Irec)
        l_shading.append(Ishd)

    # 保存结果
    save(l_image, PROJECT_DIR, 'result', 'origin')
    save(l_albedo, PROJECT_DIR, 'result', 'albedo')
    save(l_albedo, PROJECT_DIR, 'result', 'albedo')
    save(l_recon, PROJECT_DIR, 'result', 'recon')
    save(l_shading, PROJECT_DIR, 'result', 'shading')


def save(im_list, PROJECT_DIR, sub_dir, imname):
    # 每行三张, 每张图三行
    col = 3
    row = 3
    # 计算需要多少张图
    count = np.ceil(len(im_list)/float(col*row))
    for i in range(int(count)):
        plt.figure(1, figsize=(30, 30))
        plt.tight_layout()
        # 显示图像
        for index, im in enumerate(im_list[i*row*col:(i+1)*row*col]):
            plt.subplot(row, col, index+1)
            plt.imshow(im)
        # 保存图像
        plt.savefig(os.path.join(PROJECT_DIR, sub_dir, '%s_%d.png' % (imname, i)))
        plt.close()


if __name__ == '__main__':
    result_dir = os.path.join(PROJECT_DIR, 'result')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    _test()
