import numpy as np
import cv2


def show_keypoint(im, info, mode=3):
    if mode == 0:
        return im
    info = info.astype(np.int)
    if mode in (1, 3):
        l, r, t, b = info[:4].astype(np.int)
        im = cv2.rectangle(im, (l, t), (r, b), (0, 0, 255), 3)
    if mode in (2, 3):
        e1 = tuple(info[4:6])
        e2 = tuple(info[6:8])
        n = tuple(info[8:10])
        m1 = tuple(info[10:12])
        m2 = tuple(info[12:14])
        im = cv2.circle(im, e1, 3, (0, 0, 255), 3)
        im = cv2.circle(im, e2, 3, (0, 0, 255), 3)
        im = cv2.circle(im, n, 3, (0, 0, 255), 3)
        im = cv2.circle(im, m1, 3, (0, 0, 255), 3)
        im = cv2.circle(im, m2, 3, (0, 0, 255), 3)
    return im
