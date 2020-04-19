from typing import List, NamedTuple, Tuple, Union

import numpy as np

from sc_tracker.cost import calculate_fs, f_a, f_a_vec, f_c, f_c_vec, f_s, f_s_vec
from sc_tracker.partition import gating, possible_assignment_generator, subgroup_by_cluster
from sc_tracker.state import (
    DetectionState,
    DetectionStateData,
    HistogramData,
    ObjectState,
    ObjectStateData,
    StructuralConstraint,
    StructuralConstraintData,
)


class ConfigSCEA(NamedTuple):
    default_cost_d0: float = 4.0
    num_cluster_member: int = 5
    gating_threshold: float = 0.7


def cost_by_anchor(
    anchor_object_index: int,
    subgroup_member: List[int],
    possible_assignment: List[int],
    structural_constraints: List[List[StructuralConstraint]],
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    default_cost_d0: float = 4.0,
) -> float:
    """Given a assignment permutation and anchor object, calculate the assignment cost

    Args:
        anchor_object_index (int): object index used as anchor, 0 based according current subgroup
        subgroup_member (List[int]): object indices off current subgroup
        possible_assignment (List[int]): assignment list from assignment generator
        structural_constraints (List[List[StructuralConstraint]]): 2D array of structural constraint
        object_states (List[ObjectState]): list of object states
        detection_states (List[DetectionState]): list of detection states
        default_cost_d0 (float, optional): Default cost value for case detection 0. Defaults to 4.0.

    Returns:
        float: cost of assignment for given anchor
    """
    # subgroup member, object start from 0, dets start from 1 (case d_0)
    object_index = subgroup_member[anchor_object_index]
    object_anchor = object_states[object_index]
    detection_index = possible_assignment[anchor_object_index] - 1
    detection_anchor = detection_states[detection_index]
    cost_value = 0.0
    for obj_index, det_index in enumerate(possible_assignment):
        real_obj_index = subgroup_member[obj_index]
        real_det_index = det_index - 1

        # object index is equal to anchor index
        if real_obj_index == object_index:
            cost_value += __cost_anchor(object_anchor, detection_anchor)

        else:
            object_state = object_states[real_obj_index]
            detection_state = detection_states[real_det_index]
            # case for d0
            if real_det_index == -1:
                cost_value += default_cost_d0
            else:
                cost_value += __cost_non_anchor(
                    object_state=object_state,
                    detection_k=detection_anchor,
                    detection_q=detection_state,
                    sc_ij=structural_constraints[object_index][real_obj_index],
                )
    return cost_value


def cost_by_anchor_vec(
    anchor_object_index: int,
    subgroup_member: List[int],
    possible_assignment: List[int],
    structural_constraints: List[List[StructuralConstraint]],
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    default_cost_d0: float = 4.0,
) -> float:
    """Given a assignment permutation and anchor object, calculate the assignment cost

    Args:
        anchor_object_index (int): object index used as anchor, 0 based according current subgroup
        subgroup_member (List[int]): object indices off current subgroup
        possible_assignment (List[int]): assignment list from assignment generator
        structural_constraints (List[List[StructuralConstraint]]): 2D array of structural constraint
        object_states (List[ObjectState]): list of object states
        detection_states (List[DetectionState]): list of detection states
        default_cost_d0 (float, optional): Default cost value for case detection 0. Defaults to 4.0.

    Returns:
        float: cost of assignment for given anchor
    """
    # subgroup member, object start from 0, dets start from 1 (case d_0)
    object_index = subgroup_member[anchor_object_index]
    object_anchor = object_states[object_index]
    detection_index = possible_assignment[anchor_object_index] - 1
    detection_anchor = detection_states[detection_index]

    # anchor_object = []
    # anchor_detection = []
    # anchor_obj_hist = []
    # anchor_det_hist = []

    non_anchor_object = []
    non_anchor_det_q = []
    non_anchor_det_k = []
    non_anchor_obj_hist = []
    non_anchor_det_q_hist = []
    non_anchor_sc = []
    total_cost_d0 = 0.0
    cost_anchor = 0.0
    for obj_index, det_index in enumerate(possible_assignment):
        real_obj_index = subgroup_member[obj_index]
        real_det_index = det_index - 1

        # object index is equal to anchor index
        if real_obj_index == object_index:
            cost_anchor += __cost_anchor(object_anchor, detection_anchor)
            # anchor_object.append(object_anchor.state())
            # anchor_obj_hist.append(object_anchor.histogram)
            # anchor_detection.append(detection_anchor.state())
            # anchor_det_hist.append(detection_anchor.histogram)
        else:
            object_state = object_states[real_obj_index]
            detection_state = detection_states[real_det_index]
            # case for d0
            if real_det_index == -1:
                total_cost_d0 += default_cost_d0
            else:
                non_anchor_object.append(object_state.state())
                non_anchor_obj_hist.append(object_state.histogram)
                non_anchor_det_q.append(detection_state.state())
                non_anchor_det_q_hist.append(detection_state.histogram)
                non_anchor_det_k.append(detection_anchor.state())
                non_anchor_sc.append(structural_constraints[object_index][real_obj_index].state())

    # anchor_object = np.array(anchor_object)
    # anchor_detection = np.array(anchor_detection)
    # anchor_obj_hist = np.array(anchor_obj_hist)
    # anchor_det_hist = np.array(anchor_det_hist)

    non_anchor_object = np.array(non_anchor_object)
    non_anchor_det_q = np.array(non_anchor_det_q)
    non_anchor_det_k = np.array(non_anchor_det_k)
    non_anchor_obj_hist = np.array(non_anchor_obj_hist)
    non_anchor_det_q_hist = np.array(non_anchor_det_q_hist)
    non_anchor_sc = np.array(non_anchor_sc).astype(np.float64)
    if len(non_anchor_sc.shape) == 3:
        non_anchor_sc = non_anchor_sc.squeeze(2)

    total_cost = total_cost_d0 + cost_anchor
    # if len(anchor_object) != 0:
    #     cost_anchor = __cost_anchor_vec(
    #         anchor_object, anchor_detection, anchor_obj_hist, anchor_det_hist
    #     )
    #     total_cost += cost_anchor
    if len(non_anchor_object) != 0:
        cost_non_anchor = __cost_non_anchor_vec(
            non_anchor_object,
            non_anchor_det_k,
            non_anchor_det_q,
            non_anchor_obj_hist,
            non_anchor_det_q_hist,
            non_anchor_sc,
        )
        total_cost += cost_non_anchor
    return total_cost


def __cost_anchor(object_state: ObjectState, detection_state: DetectionState) -> float:
    cost_fs = f_s(object_state, detection_state)
    cost_fa = f_a(object_state, detection_state)
    return cost_fs + cost_fa


def __cost_anchor_vec(
    object_state: Union[np.ndarray, List[ObjectStateData]],
    detection_state: Union[np.ndarray, List[DetectionStateData]],
    object_hist: Union[np.ndarray, List[HistogramData]],
    detection_hist: Union[np.ndarray, List[HistogramData]],
) -> float:
    cost_fs = f_s_vec(object_state, detection_state)
    cost_fa = f_a_vec(object_hist, detection_hist)
    return (cost_fs + cost_fa).sum()


def __cost_non_anchor(
    object_state: ObjectState,
    detection_k: DetectionState,
    detection_q: DetectionState,
    sc_ij: StructuralConstraint,
) -> float:
    cost_fs = f_s(object_state, detection_q)
    cost_fa = f_a(object_state, detection_q)
    cost_fc = f_c(object_state, sc_ij, detection_k, detection_q)
    return cost_fs + cost_fa + cost_fc


def __cost_non_anchor_vec(
    object_state: Union[np.ndarray, List[ObjectStateData]],
    detection_k: Union[np.ndarray, List[DetectionStateData]],
    detection_q: Union[np.ndarray, List[DetectionStateData]],
    object_hist: Union[np.ndarray, List[HistogramData]],
    detection_q_hist: Union[np.ndarray, List[HistogramData]],
    sc_ij: Union[np.ndarray, List[StructuralConstraintData]],
):
    cost_fs = f_s_vec(object_state, detection_q)
    cost_fa = f_a_vec(object_hist, detection_q_hist)
    cost_fc = f_c_vec(object_state, detection_q, detection_k, sc_ij)
    return (cost_fs + cost_fa + cost_fc).sum()


def cost_by_possible_assignment(
    possible_assignment: List[int],
    subgroup: List[int],
    structural_constraints: List[List[StructuralConstraint]],
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    default_cost_d0: float = 4.0,
) -> float:
    anchor_count = 0
    cost = 0.0
    for object_index, detection_index in enumerate(possible_assignment):
        if detection_index == 0:
            continue
        anchor_count += 1
        cost += cost_by_anchor_vec(
            anchor_object_index=object_index,
            subgroup_member=subgroup,
            possible_assignment=possible_assignment,
            structural_constraints=structural_constraints,
            object_states=object_states,
            detection_states=detection_states,
            default_cost_d0=default_cost_d0,
        )
    if anchor_count == 0:
        return np.inf
    cost /= anchor_count
    return cost


def best_assignment_by_subgroup(
    subgroup: List[int],
    gated_assignment_matrix: np.ndarray,
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    structural_constraints: Union[np.ndarray, List[List[StructuralConstraint]]],
    default_cost_d0: float = 4.0,
) -> Tuple[np.ndarray, float]:
    """Compute the best assignment matrix for a given object subgroup and assignment matrix

    Args:
        subgroup (List[int]): subgroup contains index number of object
        gated_assignment_matrix (np.ndarray): assignment matrix from gating process
        object_states (List[ObjectState]): list of M object states
        detection_states (List[DetectionState]): list of N detection states

    Returns:
        np.ndarray: assignment matrix with shape (M, N) for M objects and N detections
    """
    min_cost = np.inf
    min_assignment_list = []
    for possible_assignment in possible_assignment_generator(gated_assignment_matrix, subgroup):
        cost = cost_by_possible_assignment(
            possible_assignment=possible_assignment,
            subgroup=subgroup,
            structural_constraints=structural_constraints,
            object_states=object_states,
            detection_states=detection_states,
            default_cost_d0=default_cost_d0,
        )
        if cost < min_cost:
            min_cost = cost
            min_assignment_list = possible_assignment.copy()
    best_assignment = np.zeros_like(gated_assignment_matrix, dtype=int)
    for obj_index, detection_index in enumerate(min_assignment_list):
        object_index = subgroup[obj_index]
        detection_index -= 1  # compensate d0
        if detection_index < 0:
            continue
        best_assignment[object_index][detection_index] = 1
    return best_assignment, min_cost


def best_assignment(
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    structural_constraints: Union[np.ndarray, List[List[StructuralConstraint]]],
    default_cost_d0: float = 4.0,
    gating_threshold: float = 0.7,
    num_cluster_member: int = 5,
) -> np.ndarray:
    assert isinstance(structural_constraints, np.ndarray), (
        "Structural constraints passed must be a numpy ndarray, "
        f"got type {type(structural_constraints)}"
    )
    fs_matrix = calculate_fs(object_states, detection_states)
    gated_assignment_matrix = gating(
        fs_matrix=fs_matrix,
        object_states=object_states,
        detection_states=detection_states,
        threshold=gating_threshold,
    )
    subgroups = subgroup_by_cluster(object_states, n_member=num_cluster_member)
    n_object = len(object_states)
    n_detection = len(detection_states)
    assignment_matrix = np.zeros((n_object, n_detection), dtype="int")
    assignment_cost = np.empty((n_object, n_detection))
    assignment_cost[:, :] = np.inf
    for subgroup in subgroups:
        current_best_assignment, current_cost = best_assignment_by_subgroup(
            subgroup=subgroup,
            gated_assignment_matrix=gated_assignment_matrix,
            object_states=object_states,
            detection_states=detection_states,
            structural_constraints=structural_constraints,
            default_cost_d0=default_cost_d0,
        )
        mask = current_best_assignment == 1
        mask_column = mask.sum(axis=0).flatten()
        mask_global_column = assignment_matrix.sum(axis=0).flatten()

        # check for same assignment
        intersection_column_mask = np.logical_and(mask_column, mask_global_column)
        if intersection_column_mask.sum() > 0:
            intersection_index = np.where(intersection_column_mask == 1)[0]
            for intersection in intersection_index:
                global_assignment_cost = assignment_cost[:, intersection].min()
                global_assignment_row = assignment_cost[:, intersection].argmin()
                if global_assignment_cost < current_cost:
                    mask[:, intersection] = False
                else:
                    assignment_matrix[global_assignment_row, intersection] = 0
        assignment_matrix[mask] = 1
        assignment_cost[mask] = current_cost
    return assignment_matrix


def get_missing_objects(assignment_matrix: np.ndarray) -> List[int]:
    summation = assignment_matrix.sum(axis=1)
    return np.argwhere(summation == 0).flatten()


def get_missing_detections(assignment_matrix: np.ndarray) -> List[int]:
    summation = assignment_matrix.sum(axis=0)
    return np.argwhere(summation == 0).flatten()
