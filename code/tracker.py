import numpy as np
from filterpy.kalman import KalmanFilter

DEV_Q = 15**2
DEV_X = 3**2
DEV_Y = 3**2
DEV_S = 15**2
DEV_W = 15**2
DEV_H = 15**

class Tracker:
    count = 0

    def __init__(self, state: np.ndarray, hist: np.ndarray):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.state = state
        self.hist = hist

        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0], 
            [0, 1, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ])
        quart_q = 0.25 * DEV_Q
        half_q = 0.5 * DEV_Q
        self.kf.Q = np.array([
            [quart_q, half_q, 0, 0, 0, 0],
            [quart_q, half_q, 0, 0, 0, 0],
            [0, 0, half_q, DEV_Q, 0, 0],
            [0, 0, half_q, DEV_Q, 0, 0],
            [0, 0, 0, 0, DEV_S, 0],
            [0, 0, 0, 0, 0, DEV_S]
        ])


