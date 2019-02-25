# coding=utf8

import numpy as np
import sys
import os
import cv2
from matplotlib import pyplot as plt

from config import *
from functions import create_shading_recon

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe


def _test():
    # set gpu mode
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    # load model and weights
    net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)

    # choose dataset
    # dat_idx = input('Please enter 1 for images with masks and 0 for images without mask: ')
    dat_idx = 1
    if dat_idx:
        # Images and masks are provided
        list_im = sorted(os.listdir('Images_mask/'))
        list_im = [im for im in list_im if im.endswith('_face.png')]
        dat_idx = 1
    elif dat_idx == 0:
        # No mask provided (Need to use your own mask).
        list_im = sorted(os.listdir('Images/'))
        list_im = [im for im in list_im if im.endswith('.png')]
        dat_idx = 0  # Uncomment to test with this mode
    else:
        sys.stderr.write('Wrong Option!')
        list_im = None

    print list_im, dat_idx

    # process every image
    for im_name in list_im:
        # read image
        if dat_idx == 1:
            # read face image as BGR format
            im = cv2.imread(os.path.join(PROJECT_DIR, 'Images_mask', im_name))
            # resize image
            im = cv2.resize(im, (M, M))
            # get mask image's name
            mask_name = im_name.replace('face', 'mask')
            # read mask image as BGR format
            Mask = cv2.imread(mask_name)
            Mask = np.float32(Mask) / 255
            mask = cv2.resize(Mask, (M, M))
        else:
            im = cv2.imread(os.path.join(PROJECT_DIR, 'Images', im_name))
            im = cv2.resize(im, (M, M))

        # prepare image
        # im=reshape(im,[size(im)]);
        im = np.float32(im)/255.0  # im=single(im)/255;
        im = np.transpose(im, [1, 0, 2])  # m_data = permute(im_data, [2, 1, 3]);
        # add by me
        im = np.transpose(im, [2, 0, 1])  # from (128, 128, 3) to (3, 128, 128)
        im = np.expand_dims(im, 0)
        print 'data shape', im.shape

        # set image data, pass images
        net.blobs['data'].data[...] = im

        # forward
        out_im = net.forward()
        n_out = out_im['Nconv0']  # normal, n_out=out_im{2};
        al_out = out_im['Aconv0']  # albedo, al_out=out_im{1};
        light_out = out_im['fc_light']  # shading, light_out=out_im{3};


        """
        light_out is a 27 dimensional vector. 9 dimension for each channel of
        RGB. For every 9 dimensional, 1st dimension is ambient illumination
        (0th order), next 3 dimension is directional (1st order), next 5
        dimension is 2nd order approximation. You can simply use 27
        dimensional feature vector as lighting representation.
        """

        print im_name
        print n_out.shape
        print al_out.shape
        print light_out.shape
        return

        # transform
        n_out2 = np.squeeze(n_out, 0)
        n_out2 = n_out2[:, :, (2, 1, 0)]
        print 'normal shape', n_out2.shape
        n_out2 = cv2.rotate(n_out2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        n_out2 = np.fliplr(n_out2)
        n_out2 = 2*n_out2-1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2**2, 3))
        n_out2 = n_out2/np.repeat(nr, (1, 1, 3))

        al_out2 = cv2.rotate(al_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        al_out2 = al_out2[:, :, (2, 1, 0)]
        al_out2 = np.fliplr(al_out2)

        # Note: n_out2, al_out2, light_out is the actual output
        Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)

        plt.figure(1)
        if dat_idx == 1:
            plt.subplot(231)
            plt.imshow(im)
            plt.title('Image')

            plt.subplot(2, 3, 2)
            plt.imshow(((1 + n_out2) / 2) * mask + (1 - mask) * np.ones(shape=(M, M, 3)))
            plt.title('Normal')

            plt.subplot(2, 3, 3)
            plt.imshow(al_out2 * mask + (1 - mask) * np.ones(shape=(M, M, 3)))
            plt.title('Albedo')

            plt.subplot(2, 3, 5)
            plt.imshow(Ishd * mask + (1 - mask) * np.ones(shape=(M, M, 3)))

            plt.title('Shading')
            plt.subplot(2, 3, 6)
            plt.imshow(Irec * mask + (1 - mask) * np.ones(shape=(M, M, 3)))
            plt.title('Recon')
        else:
            plt.subplot(231)
            plt.imshow(im)
            plt.title('Image')
            plt.subplot(232)
            plt.imshow((1+n_out2)/2)
            plt.title('Normal')
            plt.subplot(233)
            plt.imshow(al_out2)
            plt.title('Albedo')
            plt.subplot(234)
            plt.imshow(Ishd)
            plt.title('Shading')
            plt.subplot(235)
            plt.imshow(Irec)
            plt.title('Recon')

        input('Press Enter to Continue')


if __name__ == '__main__':
    _test()
