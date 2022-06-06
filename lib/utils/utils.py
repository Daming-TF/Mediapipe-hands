import math
import numpy as np


def smooth_pts(pts_last, pts, box):
    """
    smooth all landmarks
    :param pts_last: [21x2] landmarks
    :param pts: [21x2] landmarks
    :param box: [(left, top), (right, bottom)]
    :return: smoothed landmarks [21x2]
    """
    x_left, y_top = box[0]
    x_right, y_bottom = box[3]
    thresh = max(abs(x_right - x_left), abs(y_bottom - y_top)) / 30

    regions = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    pts_smoothed = np.zeros_like(pts, dtype=pts.dtype)
    for idx_region in regions:
        pts_smoothed[idx_region] = smooth_region(idx_region, pts_last, pts, thresh)
    return pts_smoothed


def smooth_region(idx_region, pts_last, pts, thresh):
    """
    smooth landmarks in a region
    :param idx_region: [n]
    :param pts_last: [21x2] landmarks
    :param pts: [21x2] landmarks
    :param thresh: num for sigmoid threshold
    :return: smoothed landmarks in a region
    """
    dist = np.mean(np.linalg.norm(pts_last[idx_region, :] - pts[idx_region, :], axis=-1))
    ratio = 1.0 - 1.0 / (1.0 + np.exp(-3.0 / thresh * (dist - thresh)))
    return ratio * pts_last[idx_region, :] + (1.0 - ratio) * pts[idx_region, :]


def coord_to_box(coords, box_factor=1.0):
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    box_c = (coord_max + coord_min) / 2
    box_size = np.max(coord_max - coord_min) * box_factor

    x_left = int(box_c[0] - box_size / 2)
    y_top = int(box_c[1] - box_size / 2)
    x_right = int(box_c[0] + box_size / 2)
    y_bottom = int(box_c[1] + box_size / 2)

    box = [(x_left, y_top), (x_right, y_bottom)]
    return box


def get_rotation_matrix(radian):
    matrix = np.array(
        [
            [math.cos(radian), math.sin(radian), 0],
            [-math.sin(radian), math.cos(radian), 0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix


def get_translation_matrix(x, y):
    matrix = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)
    return matrix


def bb_iou(box_a, box_b):
    box_a = np.array(box_a).flatten()
    box_b = np.array(box_b).flatten()

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if inter_area == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou
