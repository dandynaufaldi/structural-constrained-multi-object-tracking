import math
from typing import List

import numpy as np

from sklearn.cluster import KMeans
from state import DetectionState, ObjectState


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
    diagonal_column_vector = diagonal_vector.reshape(diagonal_vector.size, 1)
    diagonal_matrix = np.tile(diagonal_column_vector, (1, len(detection_states)))

    assert diagonal_matrix.shape == fs_matrix.shape, (
        "Dimension mismatch, diagonal matrix is %s and fs matrix is %s"
        % (diagonal_matrix.shape, fs_matrix.shape)
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
