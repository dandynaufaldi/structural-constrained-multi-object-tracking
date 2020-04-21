import math
from typing import Iterable, List, Set, Tuple

import numpy as np

from sc_tracker.state import DetectionState, ObjectState
from sklearn.cluster import KMeans


def __center_distance(
    object_states: List[ObjectState], detection_states: List[DetectionState]
) -> np.ndarray:
    distance_matrix = []
    for object_state in object_states:
        row = []
        for detection_state in detection_states:
            x = object_state.x - detection_state.x
            y = object_state.y - detection_state.y
            distance = x ** 2 + y ** 2
            row.append(distance)
        distance_matrix.append(row)
    distance_matrix = np.array(distance_matrix)
    # distance_matrix = np.sqrt(distance_matrix)
    return distance_matrix


def __diagonal(object_states: List[ObjectState]) -> np.ndarray:
    diagonals = []
    for object_state in object_states:
        diagonal = object_state.width ** 2 + object_state.height ** 2
        diagonals.append(diagonal)
    diagonals = np.array(diagonals)
    # diagonals = np.sqrt(diagonals)
    return diagonals


def gating(
    fs_matrix: np.ndarray,
    object_states: List[ObjectState],
    detection_states: List[DetectionState],
    threshold: float = 0.7,
) -> np.ndarray:
    """Remove negligible assignment from assignment matrix with shape (M,N)
    for M objects and N detections
    
    Args:
        fs_matrix (np.ndarray): Matrix of Fs cost with shape (M,N)
        object_states (List[ObjectState]): array of M object states
        detection_states (List[DetectionState]): array of N detection states
        threshold (float, optional): threshold value for Fs matrix. Defaults to 0.7.
    
    Returns:
        np.ndarray: Mask array with shape (M, N)
    """
    distance_matrix = __center_distance(object_states, detection_states)
    diagonal_vector = __diagonal(object_states)
    diagonal_column_vector = diagonal_vector.reshape(-1, 1)
    diagonal_matrix = np.tile(diagonal_column_vector, (1, len(detection_states)))

    assert diagonal_matrix.shape == fs_matrix.shape, (
        f"Dimension mismatch, diagonal matrix is {diagonal_matrix.shape} "
        f"and fs matrix is {fs_matrix.shape}"
    )
    mask = (distance_matrix < diagonal_matrix) & (np.exp(-fs_matrix) > threshold)
    return mask


def subgroup_by_cluster(object_states: List[ObjectState], n_member: int = 5) -> List[List[int]]:
    """Generate subgroup based on K-means clustering on object's position
    
    Args:
        object_states (List[ObjectState]): array of object states
        n_member (int, optional): max number of member per subgroup. Default to 5.
    
    Returns:
        List[List[int]]: 2D array, each row contains indices of objects that belong to same subgroup
    """
    n_cluster = math.ceil(len(object_states) / n_member)
    features = [[obj.x, obj.y] for obj in object_states]
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(features)
    groups = []
    for label in set(labels):
        indices = np.flatnonzero(labels == label)
        groups.append(indices.tolist())
    return groups


def __assignment_matrix_for_subgroup(
    gated_assignment_matrix: np.ndarray, subgroup: List[int]
) -> np.ndarray:
    vector_d0 = np.ones((len(gated_assignment_matrix), 1), dtype="int")
    augmented_assignment_matrix = np.hstack((vector_d0, gated_assignment_matrix))
    return augmented_assignment_matrix[subgroup]


def possible_assignment_generator(
    gated_assignment_matrix: np.ndarray, subgroup: List[int]
) -> Iterable[List[int]]:
    """Permute all possible assignment given assignment matrix from gating and subgroup

    Args:
        gated_assignment_matrix (np.ndarray): assignment matrix from gating process
        subgroup (List[int]): subgroup contains index number of object

    Yields:
        Iterable[List[int]]: assignment list
        Assignment list has lenth same as current subgroup being used and contain detection index
        assigned to an object.
        Example:
        Subgroup = [2,3,4], which is object with index 2,3, and 4 (0 based)
        Assignment list = [0, 1, 0], which map object 2 to detection 0, object 3 to detection 1
        and object 4 to detection 0. Detection 0 for mis-detected
    """
    assignment_matrix = __assignment_matrix_for_subgroup(gated_assignment_matrix, subgroup)
    n_object = len(subgroup)
    stack: List[Tuple[int, List[int], Set[int]]] = [(0, [], set())]
    while stack:
        index, array, picked = stack.pop()
        if len(array) == n_object:
            yield array
        if index == n_object:
            continue
        else:
            for i, val in enumerate(assignment_matrix[index]):
                if val == 1:
                    if i != 0 and i in picked:
                        continue
                    arr = array.copy()
                    arr.append(i)
                    pick = picked.copy()
                    pick.add(i)
                    stack.append((index + 1, arr, pick))


def possible_assignment_generator_v2(
    gated_assignment_matrix: np.ndarray, subgroup: List[int]
) -> List[List[int]]:
    """Permute all possible assignment given assignment matrix from gating and subgroup

    Args:
        gated_assignment_matrix (np.ndarray): assignment matrix from gating process
        subgroup (List[int]): subgroup contains index number of object

    Returns:
        List[List[int]]: assignment list
        Assignment list has lenth same as current subgroup being used and contain detection index
        assigned to an object.
        Example:
        Subgroup = [2,3,4], which is object with index 2,3, and 4 (0 based)
        Assignment list = [0, 1, 0], which map object 2 to detection 0, object 3 to detection 1
        and object 4 to detection 0. Detection 0 for mis-detected

    Reference:
        https://stackoverflow.com/a/35608701/13161170
    """
    assignment_matrix = __assignment_matrix_for_subgroup(gated_assignment_matrix, subgroup)
    n_object = len(subgroup)
    possible_assignment = [[] for _ in range(n_object)]
    rows, cols = assignment_matrix.nonzero()
    for object_index, detection_index in zip(rows, cols):
        possible_assignment[object_index].append(detection_index)
    possible_assignment_combination = np.array(np.meshgrid(*possible_assignment))
    possible_assignment_combination = possible_assignment_combination.T.reshape(-1, n_object)
    possible_assignment_combination = __remove_duplicate_row(
        possible_assignment_combination, excludes=[0]
    )
    return possible_assignment_combination.tolist()


def __remove_duplicate_row(matrix: np.ndarray, excludes: List[int] = None) -> np.ndarray:
    """Reference: https://stackoverflow.com/a/45136720/13161170"""
    sorted_matrix = np.sort(matrix, axis=-1)
    mask = sorted_matrix[..., 1:] != sorted_matrix[..., :-1]
    if excludes is not None:
        exclude = excludes[0]
        exclude_mask = (sorted_matrix[..., 1:] == exclude) & (sorted_matrix[..., :-1] == exclude)
        for exclude in excludes[1:]:
            temp_mask = (sorted_matrix[..., 1:] == exclude) & (sorted_matrix[..., :-1] == exclude)
            exclude_mask |= temp_mask
        mask |= exclude_mask
    return matrix[mask.all(-1)]
