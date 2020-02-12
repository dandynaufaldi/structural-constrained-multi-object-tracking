from typing import List

import numpy as np

from cost import calculate_structural_constraint, f_a, f_c, f_s
from partition import possible_assignment_generator
from state import DetectionState, ObjectState, StructuralConstraint


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


def __cost_anchor(object_state: ObjectState, detection_state: DetectionState) -> float:
    cost_fs = f_s(object_state, detection_state)
    cost_fa = f_a(object_state, detection_state)
    return cost_fs + cost_fa


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


def cost_by_possible_assignment(
    possible_assignment: List[int],
    subgroup: List[int],
    structural_constraints: List[List[StructuralConstraint]],
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
) -> float:
    anchor_count = 0
    cost = 0.0
    for object_index, detection_index in enumerate(possible_assignment):
        if detection_index == 0:
            continue
        anchor_count += 1
        cost += cost_by_anchor(
            anchor_object_index=object_index,
            subgroup_member=subgroup,
            possible_assignment=possible_assignment,
            structural_constraints=structural_constraints,
            object_states=object_states,
            detection_states=detection_states,
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
) -> np.ndarray:
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
    min_assignment_list = None
    structural_constraints = calculate_structural_constraint(object_states)
    for possible_assignment in possible_assignment_generator(gated_assignment_matrix, subgroup):
        cost = cost_by_possible_assignment(
            possible_assignment=possible_assignment,
            subgroup=subgroup,
            structural_constraints=structural_constraints,
            object_states=object_states,
            detection_states=detection_states,
        )
        if cost < min_cost:
            min_cost = cost
            min_assignment_list = possible_assignment.copy()
    best_assignment = np.zeros_like(gated_assignment_matrix)
    for obj_index, detection_index in enumerate(min_assignment_list):
        object_index = subgroup[obj_index]
        detection_index -= 1  # compensate d0
        if detection_index < 0:
            continue
        best_assignment[object_index][detection_index] = 1
    return best_assignment
