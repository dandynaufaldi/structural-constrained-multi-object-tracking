from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter

from sc_tracker.state import DetectionState, ObjectState
from scipy.linalg import block_diag

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
    INDEX_X = 0
    INDEX_Y = 1
    INDEX_V_X = 2
    INDEX_V_Y = 3
    INDEX_W = 4
    INDEX_H = 5

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
                [quart_q, 0, half_q, 0, 0, 0],
                [0, quart_q, 0, half_q, 0, 0],
                [half_q, 0, DEV_Q, 0, 0, 0],
                [0, half_q, 0, DEV_Q, 0, 0],
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

        self.__state = None
        self.__set_state()
        self.history: List[ObjectState] = [self.state]

    def update(self, detection: DetectionState):
        self.__update_state(detection)
        self.__update_histogram(detection)

        self.frame_step = detection.frame_step
        self.__set_state()
        self.history.append(self.state)

    def __update_state(self, detection: DetectionState):
        measurement = np.array([detection.x, detection.y, detection.width, detection.height])
        self.kf.update(measurement)

    def __update_histogram(self, detection: DetectionState):
        old_value = (1 - self.hist_alpha) * self.histogram
        current_value = self.hist_alpha * detection.histogram
        self.histogram = old_value + current_value

    def predict(self):
        self.kf.predict()
        self.__set_state()

    def __set_state(self):
        kf_x = self.kf.x
        object_state = ObjectState(
            x=kf_x[self.INDEX_X],
            y=kf_x[self.INDEX_Y],
            width=kf_x[self.INDEX_W],
            height=kf_x[self.INDEX_H],
            frame_step=self.frame_step,
            histogram=self.histogram,
            v_x=kf_x[self.INDEX_V_X],
            v_y=kf_x[self.INDEX_V_Y],
        )
        self.__state = object_state

    @property
    def state(self) -> ObjectState:
        return self.__state
