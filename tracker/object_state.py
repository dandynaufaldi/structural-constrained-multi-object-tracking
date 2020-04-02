import numpy as np
from filterpy.kalman import KalmanFilter

from scipy.linalg import block_diag
from state import DetectionState, ObjectState

DIM_X = 6
DIM_Z = 4
DEV_Q = 15 ** 2
DEV_X = 3 ** 2
DEV_Y = 3 ** 2
DEV_S = 15 ** 2
DEV_W = 15 ** 2
DEV_H = 15 ** 2


class ObjectStateTracker:
    counter = 0

    def __init__(self, initial_state: ObjectState):
        self.id = ObjectStateTracker.counter
        ObjectStateTracker.counter += 1

        self.kf = KalmanFilter(dim_x=DIM_X, dim_z=DIM_Z)

        F = np.eye(DIM_X)
        F[[0, 1], [2, 3]] = 1
        self.kf.F = F

        H = np.zeros((DIM_Z, DIM_X))
        H[[0, 1, 2, 3], [0, 1, 4, 5]] = 1
        self.kf.H = H

        quart_q = 0.25 * DEV_Q
        half_q = 0.5 * DEV_Q
        self.kf.Q = np.array(
            [
                [quart_q, half_q, 0, 0, 0, 0],
                [quart_q, half_q, 0, 0, 0, 0],
                [0, 0, half_q, DEV_Q, 0, 0],
                [0, 0, half_q, DEV_Q, 0, 0],
                [0, 0, 0, 0, DEV_S, 0],
                [0, 0, 0, 0, 0, DEV_S],
            ]
        )

        self.kf.R = block_diag(DEV_X, DEV_Y, DEV_W, DEV_H)
        self.kf.x = np.array(
            [
                initial_state.x,
                initial_state.y,
                initial_state.v_x,
                initial_state.v_y,
                initial_state.width,
                initial_state.height,
            ]
        )

        self.frame_step = initial_state.frame_step
        self.histogram = initial_state.histogram.copy()
        self.hist_alpha = 0.1
        self.history = [self.state]

    def update(self, detection: DetectionState):
        self.__update_state(detection)
        self.__update_histogram(detection)

        self.frame_step = detection.frame_step
        self.history.append(self.state)

    def __update_state(self, detection: DetectionState):
        measurement = np.array([detection.x, detection.y, detection.width, detection.height])
        self.kf.update(measurement)
        self.kf.predict()

    def __update_histogram(self, detection: DetectionState):
        old_value = (1 - self.hist_alpha) * self.histogram
        current_value = self.hist_alpha * detection.histogram
        self.histogram = old_value + current_value

    @property
    def state(self):
        return np.ravel(self.kf.x)
