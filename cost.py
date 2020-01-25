import math
from typing import List

import numpy as np
from numba import jit

from state import DetectionState, ObjectState, StructuralConstraint


def calculate_structural_constraint(
    object_states: List[ObjectState],
) -> List[List[StructuralConstraint]]:
    """Calculate structural constraint for every pair of objects
    
    Args:
        object_states (List[ObjectState]): array of M object states
    
    Returns:
        List[List[StructuralConstraint]]:
            2D array with shape (M, M) constains structural constraint for every pair objects
    """
    structural_constraints = [
        [None for _ in range(len(object_states))] for _ in range(len(object_states))
    ]
    for i, first_object in enumerate(object_states):
        for j, second_object in enumerate(object_states):
            if i == j:
                continue
            sc_ij = StructuralConstraint(first_object, second_object)
            sc_ji = StructuralConstraint(second_object, first_object)
            structural_constraints[i][j] = sc_ij
            structural_constraints[j][i] = sc_ji
    return structural_constraints


def calculate_fs(
    object_states: List[ObjectState], detection_states: List[DetectionState]
) -> np.ndarray:
    """Calculate Fs cost matrix given M objects and N detections
    
    Args:
        object_states (List[ObjectState]): array of M objects
        detection_states (List[DetectionState]): array of N detections
    
    Returns:
        np.ndarray: MxN Fs matrix
    """
    fs_matrix = []
    for object_state in object_states:
        fs_row = []
        for detection_state in detection_states:
            fs = __f_s(
                object_state.height,
                object_state.width,
                detection_state.height,
                detection_state.width,
            )
            fs_row.append(fs)
        fs_matrix.append(fs_row)
    return np.array(fs_matrix)


@jit
def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    bbox1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    bbox2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    o = wh / (bbox1 + bbox2 - wh)
    return o


def __f_s(h_obj: float, w_obj: float, h_det: float, w_det: float) -> float:
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
