from typing import List

from cost import f_a, f_c, f_s
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
    cost /= anchor_count
    return cost
