import numpy as np
from filterpy.kalman import KalmanFilter

from state import DetectionState, ObjectState


class ObjectStateTracker:
    def __init__(self, initial_state: ObjectState):
        dim_x = 6
        dim_z = 4
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        F = np.eye(dim_x)
        F[[0, 1], [2, 3]] = 1
        self.kf.F = F

        H = np.zeros((dim_z, dim_x))
        H[:2, :2] = np.eye(2)
        H[:2, 4:] = np.eye(2)
        self.kf.H = H

    def update(self, detection: DetectionState):
        pass

    def __update_state(self, detection: DetectionState):
        pass

    def __update_histogram(self, detection: DetectionState):
        pass

    @property
    def state(self):
        pass

    @property
    def histogram(self):
        pass


class StructuralConstraintTracker:
    def __init__(self):
        self.kf = KalmanFilter()

    def update_well_tracked(self):
        pass

    def update_missing(self):
        pass
