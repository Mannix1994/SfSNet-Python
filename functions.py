# coding=utf8
import numpy as np
import sys


def create_shading_recon(n_out2, al_out2, light_out):
    """
    :type n_out2: np.ndarray
    :type al_out2: np.ndarray
    :type light_out: np.ndarray
    :return:
    """
    M = n_out2.shape[0]
    No1 = np.reshape(n_out2, (M * M, 3))
    tex1 = np.reshape(al_out2, (M * M, 3))

    la = lambertian_attenuation(3)
    HN1 = normal_harmonics(No1.T, la)
    HS1r = HN1 * light_out[0:8]
    HS1g = HN1 * light_out[9:17]
    HS1b = HN1 * light_out[18:26]

    HS1 = np.zeros(shape=(M, M, 3))
    HS1[:, :, 1]=np.reshape(HS1r, (M, M))
    HS1[:, :, 2]=np.reshape(HS1g, (M, M))
    HS1[:, :, 3]=np.reshape(HS1b, (M, M))
    Tex1 = np.reshape(tex1, (M, M, 3)) * HS1

    IRen0 = Tex1
    Shd = (200 / 255) * HS1  # 200 is added instead of 255 so that not to scale the shading to all white
    Ishd0 = Shd
    return [IRen0, Ishd0]


def lambertian_attenuation(n):
    # a = [.8862; 1.0233; .4954];
    a = np.pi * [1, 2 / 3, .25]
    if n > 3:
        sys.stderr.write('didnt record more than 3 attenuations')
        exit(-1)
    o = a[0:n - 1]
    return o


def normal_harmonics(N, att):
    """
    Return the harmonics evaluated at surface normals N, attenuated by att.
    :param N:
    :param att:
    :return:

    Normals can be scaled surface normals, in which case value of each
    harmonic at each point is scaled by albedo.
    Harmonics written as polynomials
    0,0    1/sqrt(4*pi)
    1,0    z*sqrt(3/(4*pi))
    1,1e    x*sqrt(3/(4*pi))
    1,1o    y*sqrt(3/(4*pi))
    2,0   (2*z.^2 - x.^2 - y.^2)/2 * sqrt(5/(4*pi))
    2,1e  x*z * 3*sqrt(5/(12*pi))
    2,1o  y*z * 3*sqrt(5/(12*pi))
    2,2e  (x.^2-y.^2) * 3*sqrt(5/(48*pi))
    2,2o  x*y * 3*sqrt(5/(12*pi))
    """
    xs = N[1, :].T
    ys = N[2, :].T
    zs = N[3, :].T
    a = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    denom = (a == 0) + a
    # %x = xs./a; y = ys./a; z = zs./a;
    x = xs / denom
    y = ys / denom
    z = zs / denom

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    H1 = att[0] * (1 / np.sqrt(4 * np.pi)) * a
    H2 = att[1] * (np.sqrt(3 / (4 * np.pi))) * zs
    H3 = att[1] * (np.sqrt(3 / (4 * np.pi))) * xs
    H4 = att[1] * (np.sqrt(3 / (4 * np.pi))) * ys
    H5 = att[2] * (1 / 2.0) * (np.sqrt(5 / (4 * np.pi))) * ((2 * z2 - x2 - y2) * a)
    H6 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xz * a)
    H7 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (yz * a)
    H8 = att[2] * (3 * np.sqrt(5 / (48 * np.pi))) * ((x2 - y2) * a)
    H9 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xy * a)
    H = [H1, H2, H3, H4, H5, H6, H7, H8, H9]
    return H
