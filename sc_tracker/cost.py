import math
from typing import List

import numpy as np

from sc_tracker.state import DetectionState, ObjectState, StructuralConstraint


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
            sc_ij = StructuralConstraint.create(first_object, second_object)
            sc_ji = StructuralConstraint.create(second_object, first_object)
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
            fs = f_s(object_state, detection_state)
            fs_row.append(fs)
        fs_matrix.append(fs_row)
    return np.array(fs_matrix)


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


def f_s(object_state: ObjectState, detection_state: DetectionState) -> float:
    h_obj, w_obj = object_state.height, object_state.width
    h_det, w_det = detection_state.height, detection_state.width
    h = abs(h_obj - h_det) / (2 * (h_obj + h_det))
    w = abs(w_obj - w_det) / (2 * (w_obj + w_det))
    return -1 * math.log(1 - h - w)


def get_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    histogram, _ = np.histogram(img, bins=bins)
    return histogram


def f_a(object_state: ObjectState, detection_state: DetectionState) -> float:
    hist_obj = object_state.histogram
    hist_det = detection_state.histogram
    root = np.sqrt(hist_obj * hist_det)
    summation = root.sum()
    return -np.log(summation)


def f_c(
    object_state: ObjectState,
    sc_ij: StructuralConstraint,
    detection_k: DetectionState,
    detection_q: DetectionState,
) -> float:
    s_jk_base = np.array([detection_k.x, detection_k.y, 0, 0])
    s_jk_add = np.array([sc_ij.delta_x, sc_ij.delta_y, object_state.width, object_state.height])
    s_jk = s_jk_base + s_jk_add

    # convert to [x1, y1, x2, y2]
    det_q = np.array([detection_q.x, detection_q.y, detection_q.width, detection_q.height])
    det_q[2:] = det_q[:2] + det_q[2:]
    s_jk[2:] = s_jk[:2] + s_jk[2:]

    iou_score = iou(s_jk, det_q)
    if iou_score == 0.0:
        return np.inf
    cost = -np.log(iou_score)
    return cost


def f_r(
    object_state: ObjectState, detection_state: DetectionState, match_gamma: ObjectState
) -> float:
    s_i_gamma_base = np.array([match_gamma.x, match_gamma.y, 0, 0])
    sc_object_gamma = StructuralConstraint.create(object_state, match_gamma)
    s_i_gamma_add = np.array(
        [sc_object_gamma.delta_x, sc_object_gamma.delta_y, object_state.width, object_state.height]
    )
    s_i_gamma = s_i_gamma_base + s_i_gamma_add

    # convert to [x1, y1, x2, y2]
    det_q = np.array(
        [detection_state.x, detection_state.y, detection_state.width, detection_state.height]
    )
    det_q[2:] = det_q[:2] + det_q[2:]
    s_i_gamma[2:] = s_i_gamma[:2] + s_i_gamma[2:]

    iou_score = iou(s_i_gamma, det_q)
    if iou_score == 0.0:
        return np.inf
    cost = -np.log(iou_score)
    return cost


if __name__ == "__main__":
    pass
