# coding=utf8

from functions import *


def _test_inpolygon():
    xc = np.array([-4, 0, 4, 0])
    yc = np.array([0, 4, 0, -4])
    X = np.array([0, 1, 3.5, 4, 5])
    Y = np.array([0, 1, 0, 0, 0])

    _in, _on = inpolygon(X, Y, xc, yc)
    assert (_in == np.array([[True, True, True, True, False]])).all()
    assert (_on == np.array([[False, False, False, True, False]])).all()


def _create_fiducials():
    y_up = np.array([10] * 17)
    y_right = np.arange(11, 28, 1, dtype=np.int)
    y_buttom = np.array([28] * 17)
    y_left = np.flip(np.arange(11, 28, 1, dtype=np.int))
    x_up = y_right.copy()
    x_right = y_buttom.copy()
    x_buttom = y_left.copy()
    x_left = x_up.copy()
    x = np.hstack([x_up, x_right, x_buttom, x_left])
    y = np.hstack([y_up, y_right, y_buttom, y_left])

    fiducials = np.vstack([x, y])

    return fiducials


def _test_create_mask_fiducial():
    fiducials = _create_fiducials()
    Image = np.ones((128, 128, 3))
    mask = create_mask_fiducial(fiducials, Image)
    import cv2
    cv2.imshow("mask", mask)
    cv2.waitKey(0)


if __name__ == '__main__':
    _test_inpolygon()
    _test_create_mask_fiducial()
else:
    raise RuntimeError('func_test.py shouldn\'t be imported')
