"""Core Tracking module

This module contain main tracking system consists of SCEA, SCOR, and state management.
"""
from typing import List, Tuple

import numpy as np

import sc_tracker.scea as scea
import sc_tracker.scor as scor
from sc_tracker.cost import calculate_structural_constraint
from sc_tracker.state import DetectionState, ObjectState, StructuralConstraint
from sc_tracker.tracker.object_state import ObjectStateTracker
from sc_tracker.tracker.structural_constraint import StructuralConstraintTracker


def reset_index(source: np.array, deleted_index: np.ndarray) -> np.ndarray:
    combined = np.concatenate((source, deleted_index))
    if not combined:
        return source
    max_value = np.max([source, deleted_index])

    mask = np.zeros(max_value + 1, dtype="bool")
    mask[source] = 1
    mask[deleted_index] = 0

    complete_mask = np.ones_like(mask, dtype="bool")
    complete_mask[deleted_index] = 0
    prefix_sum = np.cumsum(complete_mask) - 1

    return prefix_sum[mask]


class Tracker:
    def __init__(self, max_age: int, scea_config: scea.ConfigSCEA, scor_config: scor.ConfigSCOR):
        self.__max_age = max_age
        self.__scea_config = scea_config
        self.__scor_config = scor_config
        self.__object_states: List[ObjectState] = []
        self.__structural_constraints: List[List[StructuralConstraint]] = []
        self.__object_trackers: List[ObjectStateTracker] = []
        self.__sc_trackers: List[List[StructuralConstraintTracker]] = []
        self.__well_tracked_indexes: List[int] = []
        self.__missing_indexes: List[int] = []
        self.__last_updates: List[int] = []
        self.__first = True

    def update(self, detections: List[DetectionState], current_frame_step: int):
        if not isinstance(detections, np.ndarray):
            detections = np.array(detections)
        if self.__first:
            self.__update_first_time(detections, current_frame_step)
        else:
            self.__update_routine(detections, current_frame_step)

    def __update_first_time(self, detections: List[DetectionState], current_frame_step: int):
        self.__object_states = np.array(
            [ObjectState.from_detection(detection) for detection in detections], dtype=object
        )
        self.__object_trackers = np.array(
            [ObjectStateTracker(object_state) for object_state in self.__object_states],
            dtype=object,
        )

        structural_constraints = calculate_structural_constraint(self.__object_states)
        if structural_constraints == []:
            self.__structural_constraints = np.array([[]], dtype=object)
        else:
            self.__structural_constraints = np.array(structural_constraints, dtype=object)
        sc_trackers = [
            [None for _ in range(len(self.__object_states))]
            for _ in range(len(self.__object_states))
        ]
        for i, row in enumerate(self.__structural_constraints):
            for j, structural_constraint in enumerate(row):
                if i == j:
                    continue
                sc_trackers[i][j] = StructuralConstraintTracker(structural_constraint)
        if structural_constraints == []:
            self.__sc_trackers = np.array([[]], dtype=object)
        else:
            self.__sc_trackers = np.array(sc_trackers, dtype=object)

        self.__well_tracked_indexes = np.arange(len(self.__object_states), dtype=int)
        self.__missing_indexes = np.array([], dtype=int)
        self.__last_updates = np.array([current_frame_step] * len(detections), dtype=int)
        self.__first = False

    def __update_routine(self, detections: List[DetectionState], current_frame_step: int):
        if not detections:
            self.__missing_indexes = np.concatenate(
                (self.__missing_indexes, self.__well_tracked_indexes)
            )
            self.__well_tracked_indexes = np.array([], dtype=int)
            self.__update_last_updates(current_frame_step=current_frame_step)
            self.__remove_missing_objects(current_frame_step=current_frame_step)
            return
        (
            scea_tracked_object_index,
            scea_assigned_detection_index,
            scea_unassigned_detections_index,
        ) = self.__process_scea(detections)

        unassigned_detection_states = detections[scea_unassigned_detections_index]
        (
            scor_tracked_object_index,
            scor_assigned_detection_index,
            scor_unassigned_detections_index,
        ) = self.__process_scor(unassigned_detection_states)

        tracked_object_indexes = np.concatenate(
            (scea_tracked_object_index, scor_tracked_object_index)
        )
        assigned_detection_indexes = np.concatenate(
            (scea_assigned_detection_index, scor_assigned_detection_index)
        )

        self.__update_object_trackers(
            tracked_object_indexes=tracked_object_indexes,
            assigned_detection_indexes=assigned_detection_indexes,
            detections=detections,
        )

        self.__update_sc_trackers(
            tracked_object_indexes=tracked_object_indexes,
            assigned_detection_indexes=assigned_detection_indexes,
            detections=detections,
        )

        self.__update_last_updates(current_frame_step=current_frame_step)
        self.__remove_missing_objects(current_frame_step=current_frame_step)
        self.__create_new_well_tracked_objects(detections[scor_unassigned_detections_index])

    def __process_scea(
        self, detections: List[DetectionState]
    ) -> Tuple[List[int], List[int], List[int]]:
        object_states = self.__object_states[self.__well_tracked_indexes]
        structural_constraints = self.__structural_constraints[self.__well_tracked_indexes, :]
        structural_constraints = structural_constraints[:, self.__well_tracked_indexes]

        assignment_matrix = scea.best_assignment(
            object_states=object_states,
            detection_states=detections,
            structural_constraints=structural_constraints,
            default_cost_d0=self.__scea_config.default_cost_d0,
            gating_threshold=self.__scea_config.gating_threshold,
            num_cluster_member=self.__scea_config.num_cluster_member,
        )
        tracked_object_indexes, assigned_detection_indexes = np.where(assignment_matrix == 1)

        self.__update_object_from_assigned_detection(
            detections=detections,
            tracked_object_indexes=tracked_object_indexes,
            assigned_detection_indexes=assigned_detection_indexes,
        )

        actual_tracked_object_index = self.__well_tracked_indexes[tracked_object_indexes]

        unassigned_detections_index = scea.get_missing_detections(assignment_matrix)
        relative_missing_objects_index = scea.get_missing_objects(assignment_matrix)
        missing_object_index = self.__well_tracked_indexes[relative_missing_objects_index]

        # update index well-tracked object
        self.__missing_indexes = np.concatenate((self.__missing_indexes, missing_object_index))
        self.__missing_indexes = self.__missing_indexes.astype("int32")
        self.__well_tracked_indexes = actual_tracked_object_index

        return (
            actual_tracked_object_index,
            assigned_detection_indexes,
            unassigned_detections_index,
        )

    def __process_scor(
        self, unassigned_detections: List[DetectionState]
    ) -> Tuple[List[int], List[int], List[int]]:
        missing_objects = self.__object_states[self.__missing_indexes]
        updated_objects = self.__object_states[self.__well_tracked_indexes]
        assignment_matrix = scor.best_assignment(
            missing_objects=missing_objects,
            unassigned_detections=unassigned_detections,
            updated_objects=updated_objects,
            default_cost_d0=self.__scor_config.default_cost_d0,
        )
        tracked_object_indexes, assigned_detection_indexes = np.where(assignment_matrix == 1)

        actual_tracked_object_index = self.__missing_indexes[tracked_object_indexes]

        unassigned_detections_index = scea.get_missing_detections(assignment_matrix)
        relative_missing_objects_index = scea.get_missing_objects(assignment_matrix)
        missing_object_index = self.__missing_indexes[relative_missing_objects_index]

        # update index well-tracked object
        self.__well_tracked_indexes = np.concatenate(
            (self.__well_tracked_indexes, actual_tracked_object_index)
        )
        self.__well_tracked_indexes = self.__well_tracked_indexes.astype("int32")
        self.__missing_indexes = missing_object_index

        return (
            actual_tracked_object_index,
            assigned_detection_indexes,
            unassigned_detections_index,
        )

    def __update_object_from_assigned_detection(
        self,
        detections: List[DetectionState],
        tracked_object_indexes: List[int],
        assigned_detection_indexes: List[int],
    ):
        object_states = self.__object_states[self.__well_tracked_indexes]
        for object_idx, detection_idx in zip(tracked_object_indexes, assigned_detection_indexes):
            detection_state = detections[detection_idx]
            object_state = object_states[object_idx]
            object_state.update_from_detection(detection_state)

    def __update_object_trackers(
        self,
        tracked_object_indexes: List[int],
        assigned_detection_indexes: List[int],
        detections: List[DetectionState],
    ):
        for object_index, detection_index in zip(
            tracked_object_indexes, assigned_detection_indexes
        ):
            detection = detections[detection_index]
            tracker = self.__object_trackers[object_index]
            tracker.update(detection)
            object_state = self.__object_states[object_index]
            object_state.update_from_object_state(tracker.state)

    def __update_sc_trackers(
        self,
        tracked_object_indexes: List[int],
        assigned_detection_indexes: List[int],
        detections: List[DetectionState],
    ):
        # update well-tracked
        objects_from_detections = [
            ObjectState.from_detection(detection) for detection in detections
        ]
        for i, object_index_i in enumerate(tracked_object_indexes):
            for j, object_index_j in enumerate(tracked_object_indexes):
                if i == j:
                    continue
                object_i = objects_from_detections[i]
                object_j = objects_from_detections[j]
                sc = StructuralConstraint.create(object_i, object_j)
                tracker = self.__sc_trackers[object_index_i][object_index_j]
                tracker.update(sc)
                saved_sc = self.__structural_constraints[object_index_i][object_index_j]
                saved_sc.update_from_sc(tracker.state)

        # update one or both missing
        mask = np.ix_(tracked_object_indexes, tracked_object_indexes)
        bool_mask = np.ones_like(self.__sc_trackers, dtype="bool")
        bool_mask[mask] = 0
        missing_sc_trackers = self.__sc_trackers[bool_mask]
        missing_sc = self.__structural_constraints[bool_mask]
        for tracker, sc in zip(missing_sc_trackers, missing_sc):
            if tracker:
                tracker.update()
                sc.update_from_sc(tracker.state)

    def __update_last_updates(self, current_frame_step: int):
        self.__last_updates[self.__well_tracked_indexes] = current_frame_step

    def __remove_missing_objects(self, current_frame_step: int):
        diff = current_frame_step - self.__last_updates
        removed_index = np.where(diff > self.__max_age)[0].flatten()
        mask = np.ones(self.__object_states.shape, bool)
        mask[removed_index] = 0

        self.__object_states = self.__object_states[mask]
        self.__object_trackers = self.__object_trackers[mask]
        self.__last_updates = self.__last_updates[mask]
        self.__structural_constraints = self.__structural_constraints[mask, :]
        self.__structural_constraints = self.__structural_constraints[:, mask]
        self.__sc_trackers = self.__sc_trackers[mask, :]
        self.__sc_trackers = self.__sc_trackers[:, mask]

        self.__well_tracked_indexes = reset_index(self.__well_tracked_indexes, removed_index)
        self.__missing_indexes = reset_index(self.__missing_indexes, removed_index)

    def __create_new_well_tracked_objects(self, unassigned_detections: List[DetectionState]):
        n_new_objects = len(unassigned_detections)
        new_object_states = [
            ObjectState.from_detection(detection) for detection in unassigned_detections
        ]
        new_object_trackers = [
            ObjectStateTracker(object_state) for object_state in new_object_states
        ]
        new_object_index = np.arange(n_new_objects) + len(self.__object_states)
        self.__well_tracked_indexes = np.concatenate(
            (self.__well_tracked_indexes, new_object_index)
        )
        self.__object_states = np.concatenate((self.__object_states, new_object_states))
        self.__object_trackers = np.concatenate((self.__object_trackers, new_object_trackers))

        sc_old_and_new = []
        for new_object in new_object_states:
            row = []
            for old_object in self.__object_states:
                row.append(StructuralConstraint.create(new_object, old_object))
            sc_old_and_new.append(row)
        sc_new_only = calculate_structural_constraint(new_object_states)
        sc_new = np.concatenate((sc_old_and_new, sc_new_only), axis=1)
        sc_tracker_new = []
        for row in sc_new:
            temp = []
            for sc in row:
                tracker = None
                if sc:
                    tracker = StructuralConstraintTracker(sc)
                temp.append(tracker)
            sc_tracker_new.append(temp)

        self.__structural_constraints = np.pad(
            self.__structural_constraints, (0, n_new_objects), "constant", constant_values=None,
        )
        self.__structural_constraints[-n_new_objects:, :] = sc_new
        self.__structural_constraints[:, -n_new_objects:] = np.transpose(sc_new)
        self.__sc_trackers = np.pad(
            self.__sc_trackers, (0, n_new_objects), "constant", constant_values=None,
        )
        self.__sc_trackers[-n_new_objects:, :] = sc_tracker_new
        self.__sc_trackers[:, -n_new_objects:] = np.transpose(sc_tracker_new)

    @property
    def tracked_objects(self):
        return self.__object_states[self.__well_tracked_indexes]
