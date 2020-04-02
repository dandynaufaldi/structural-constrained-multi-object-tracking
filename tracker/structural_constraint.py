from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from scipy.linalg import block_diag
from state import StructuralConstraint

DIM_X = 4
DIM_Z = 2
DEV_SC = 1 ** 2
DEV_X = 3 ** 2
DEV_Y = 3 ** 2


class StructuralConstraintTracker:
    def __init__(self, initial_state: StructuralConstraint):
        self.kf = KalmanFilter(dim_x=DIM_X, dim_z=DIM_Z)

        F = np.eye(DIM_X)
        F[[0, 1], [2, 3]] = 1
        self.kf.F = F

        H = np.zeros((DIM_Z, DIM_X))
        H[[0, 1], [0, 1]] = 1
        self.kf.H = H

        quart_sc = 0.25 * DEV_SC
        half_sc = 0.5 * DEV_SC
        self.kf.Q = np.array(
            [
                [quart_sc, half_sc, 0, 0],
                [quart_sc, half_sc, 0, 0],
                [0, 0, half_sc, DEV_SC],
                [0, 0, half_sc, DEV_SC],
            ]
        )

        self.kf.R = block_diag(DEV_X, DEV_Y)
        self.kf.x = np.array(
            [
                initial_state.delta_x,
                initial_state.delta_y,
                initial_state.delta_v_x,
                initial_state.delta_v_y,
            ]
        )
        self.history = [self.state]

    def update(self, sc: Optional[StructuralConstraint] = None):
        if sc:
            self.__update_well_tracked(sc)
        else:
            self.__update_missing()
        self.history.append(self.state)

    def __update_well_tracked(self, sc: StructuralConstraint):
        measurement = np.array([sc.delta_x, sc.delta_y])
        self.kf.update(measurement)
        self.kf.predict()

    def __update_missing(self):
        column_vector_x = self.kf.x.reshape((-1, 1))
        self.kf.x = np.dot(self.kf.F, column_vector_x)

    @property
    def state(self):
        return np.ravel(self.kf.x)
