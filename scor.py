import math
from typing import List

import numpy as np

import cost
from scipy.optimize import linear_sum_assignment
from state import DetectionState, ObjectState


def velocity_resultant(object_i: ObjectState, object_j: ObjectState) -> float:
    delta_v_x = object_i.v_x - object_j.v_x
    delta_v_y = object_i.v_y - object_j.v_y
    # return math.sqrt(delta_v_x ** 2 + delta_v_y ** 2)
    return delta_v_x ** 2 + delta_v_y ** 2


def matching_s_gamma(
    updated_objects: List[ObjectState], missing_objects: List[ObjectState]
) -> List[ObjectState]:
    s_gamma = []
    for missing in missing_objects:
        min_resultant = math.inf
        min_state = None
        for well_tracked in updated_objects:
            resultant = velocity_resultant(missing, well_tracked)
            if resultant < min_resultant:
                min_resultant = resultant
                min_state = well_tracked
        s_gamma.append(min_state)
    return s_gamma


def assignment_cost(
    missing_object: ObjectState, unassigned_detection: DetectionState, match_s: ObjectState
) -> float:
    f_a = cost.f_a(missing_object, unassigned_detection)
    f_s = cost.f_s(missing_object, unassigned_detection)
    f_r = cost.f_r(missing_object, unassigned_detection, match_s)
    value = f_a + f_s + f_r
    return value


def calculate_assignment_cost_matrix(
    missing_objects: List[ObjectState],
    unassigned_detections: List[DetectionState],
    updated_objects: List[ObjectState],
    default_cost_d0: float = 4.0,
) -> np.ndarray:
    match_s_gamma = matching_s_gamma(
        updated_objects=updated_objects, missing_objects=missing_objects
    )
    costs = []
    for missing_object, match_s in zip(missing_objects, match_s_gamma):
        row = []
        for unassigned_detection in unassigned_detections:
            value = assignment_cost(missing_object, unassigned_detection, match_s)
            row.append(value)
        costs.append(row)
    costs = np.array(costs)

    n_objects = len(missing_objects)
    costs_d0 = np.eye(n_objects)
    costs_d0[costs_d0 == 0.0] = np.inf
    costs_d0[costs_d0 == 1.0] = default_cost_d0
    cost_augmented = np.hstack((costs, costs_d0))
    return cost_augmented


def best_assignment(
    missing_objects: List[ObjectState],
    unassigned_detections: List[DetectionState],
    updated_objects: List[ObjectState],
    default_cost_d0: float = 4.0,
) -> np.ndarray:
    assignment_cost_matrix = calculate_assignment_cost_matrix(
        missing_objects=missing_objects,
        unassigned_detections=unassigned_detections,
        updated_objects=updated_objects,
        default_cost_d0=default_cost_d0,
    )
    max_val = np.max(assignment_cost_matrix[assignment_cost_matrix != np.inf])
    assignment_cost_matrix[assignment_cost_matrix == np.inf] = max_val * 2
    rows, cols = linear_sum_assignment(assignment_cost_matrix)

    n_unassigned_det = len(unassigned_detections)
    n_missing_objects = len(missing_objects)

    assignment_matrix = np.zeros(
        (n_missing_objects, n_unassigned_det + n_missing_objects), dtype="int"
    )
    assignment_matrix[rows, cols] = 1

    # exclude column for d0
    assignment_matrix = assignment_matrix[:, :n_unassigned_det]
    return assignment_matrix
