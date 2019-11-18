import math

import numpy as np

from numba import jit


@jit
def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) *
              (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def f_s(h_obj: float, w_obj: float, h_det: float, w_det: float) -> float:
    h = abs(h_obj - h_det) / (2 * (h_obj + h_det))
    w = abs(w_obj - w_det) / (2 * (w_obj + w_det))
    return -1 * math.log(1 - h - w)


def get_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    histogram, _ = np.histogram(img, bins=bins)
    return histogram


def f_a(hist_obj: np.ndarray, hist_det: np.ndarray) -> float:
    root = np.sqrt(hist_obj * hist_det)
    summation = np.sum(root)
    return -np.log(summation)


def f_c(state_j: np.ndarray, e_ij: np.ndarray, det_k: np.ndarray, det_q: np.ndarray) -> float:
    """
    Args:
        state_j (np.ndarray): [x_j, y_j, w_j. h_j]
        e_ij (np.ndarray): [x_ij, y_ij, v_x_ij. v_y_ij]
        det_k (np.ndarray): [x_k, y_k, w_k. h_k]
        det_q (np.ndarray): [x_q, y_q, w_q. h_q]
    
    Returns:
        float: f_c cost
    """
    s_jk_base = np.array([det_k[0], det_k[1], 0, 0])
    s_jk_add = np.array([e_ij[0], e_ij[1], state_j[2], state_j[3]])
    s_jk = s_jk_base + s_jk_add

    # convert to [x1, y1, x2, y2]
    det_q = np.copy(det_q)
    det_q[2:] = det_q[:2] + det_q[2:]
    s_jk[2:] = s_jk[:2] + s_jk[2:]

    iou_score = iou(s_jk, det_q)
    cost = -np.log(iou_score)
    return cost


if __name__ == "__main__":
    pass
