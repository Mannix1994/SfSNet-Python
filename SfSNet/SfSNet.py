# coding=utf8

import numpy as np
import sys
import os
import shutil
import cv2
from matplotlib import pyplot as plt

from config import CAFFE_ROOT, M
from functions import create_shading_recon

# the two lines add pycaffe support
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe
from mask import MaskGenerator


class SfSNet:
    def __init__(self, model, weights, gpu_id=0,
                 landmark_path='shape_predictor_68_face_landmarks.dat'):
        """
        init SfSNet
        :param model: model's path
        :param weights: weights's path
        :param gpu_id: gpu id you want use, if you don't compile caffe
        with CUDA support, set gpu_id to None.
        :param landmark_path the path of pretrained key points weight used
        in dlib. it could be download from:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        if gpu_id:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            caffe.set_mode_cpu()

        # load model and weights
        self.net = caffe.Net(model, weights, caffe.TEST)

        # define a mask generator
        self.mg = MaskGenerator(landmark_path)

    def forward(self, image, show=False):
        """
        compute albedo, normal, shading, reconstruction of image
        :param image: the image you want to process
        :return: cropped_face, albedo, normal, shading
        """
        mask, im = self.mg.align(image, crop_size=(M, M))
        if show:
            cv2.imshow("mask", mask)
            cv2.imshow("image", im)
            cv2.waitKey(50)

        # prepare image
        # im=reshape(im,[size(im)]);
        im = np.float32(im) / 255.0  # im=single(im)/255;
        # im = np.transpose(im, [1, 0, 2])  # m_data = permute(im_data, [2, 1, 3]); switch width and height

        # -----------add by wang-------------
        im = np.transpose(im, [2, 0, 1])  # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.expand_dims(im, 0)
        # print 'im shape', im.shape
        # -----------end---------------------

        # set image data, pass images
        self.net.blobs['data'].data[...] = im

        # forward
        out_im = self.net.forward()
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
        n_out2 = 2 * n_out2 - 1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
        nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
        # print 'nr shape', nr.shape

        al_out2 = cv2.rotate(al_out, cv2.ROTATE_90_CLOCKWISE)
        al_out2 = al_out2[:, :, (2, 1, 0)]
        al_out2 = np.fliplr(al_out2)

        # Note: n_out2, al_out2, light_out is the actual output
        Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)

        diff = (mask / 255)

        n_out2 = n_out2 * diff
        al_out2 = al_out2 * diff
        Ishd = Ishd * diff
        Irec = Irec * diff

        Ishd = np.float32(Ishd)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)
        Ishd = cv2.cvtColor(Ishd, cv2.COLOR_GRAY2RGB)
        return im, n_out2, al_out2, Irec, Ishd
