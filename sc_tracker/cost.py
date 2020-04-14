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


def iou(bb_test: List[float], bb_gt: List[float]) -> float:
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
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


def f_a(object_state: ObjectState, detection_state: DetectionState) -> float:
    hist_obj = object_state.histogram
    hist_det = detection_state.histogram
    root = np.sqrt(hist_obj * hist_det)
    summation = root.sum()
    return -math.log(summation)


def f_c(
    object_state: ObjectState,
    sc_ij: StructuralConstraint,
    detection_k: DetectionState,
    detection_q: DetectionState,
) -> float:
    x = detection_k.x - detection_k.width / 2 + sc_ij.delta_x
    y = detection_k.y - detection_k.height / 2 + sc_ij.delta_y
    s_jk = [x, y, x + object_state.width, y + object_state.height]

    det_q = [
        detection_q.x - detection_q.width / 2,
        detection_q.y - detection_q.height / 2,
        detection_q.x + detection_q.width / 2,
        detection_q.y + detection_q.height / 2,
    ]

    iou_score = iou(s_jk, det_q)
    if iou_score == 0.0:
        return np.inf
    cost = -math.log(iou_score)
    return cost


def f_r(
    object_state: ObjectState, detection_state: DetectionState, match_gamma: ObjectState
) -> float:
    sc_object_gamma = StructuralConstraint.create(object_state, match_gamma)
    x = match_gamma.x - match_gamma.width / 2 + sc_object_gamma.delta_x
    y = match_gamma.y - match_gamma.height / 2 + sc_object_gamma.delta_y
    s_i_gamma = [x, y, x + object_state.width, y + object_state.height]

    det_q = [
        detection_state.x - detection_state.width / 2,
        detection_state.y - detection_state.height / 2,
        detection_state.x + detection_state.width / 2,
        detection_state.y + detection_state.height / 2,
    ]

    iou_score = iou(s_i_gamma, det_q)
    if iou_score == 0.0:
        return np.inf
    cost = -math.log(iou_score)
    return cost
