import numpy as np
from filterpy.kalman import KalmanFilter

DEV_Q = 15 ** 2
DEV_X = 3 ** 2
DEV_Y = 3 ** 2
DEV_S = 15 ** 2
DEV_W = 15 ** 2
DEV_H = 15 ** 2


def convert_bbox_to_z(bbox: np.ndarray):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float = None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape(
            (1, 4)
        )
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def diagonal_matrix(value: np.ndarray) -> np.ndarray:
    """Create a zeros matrix with it's diugonal filled by values
    
    Arguments:
        value {np.ndarray} -- 1-D array
    
    Returns:
        np.ndarray -- zeros matrix with it's diugonal filled by values
    """
    assert isinstance(value, np.ndarray)
    assert value.ndim == 1

    size = len(value)
    arr = np.zeros((size, size))
    diagonal_idx = np.diag_indices(size)
    arr[diagonal_idx] = value
    return arr


class Tracker:
    count = 0

    def __init__(self, state: np.ndarray, hist: np.ndarray, dim_x: int = 6, dim_z: int = 4):
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        self.state = state
        self.hist = hist

        self.kf.F = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
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
        H = np.zeros((dim_z, dim_x))
        H[:2, :2] = np.eye(2)
        H[:2, 4:] = np.eye(2)
        self.kf.H = H

        self.kf.R = diagonal_matrix(np.array([DEV_X, DEV_Y, DEV_W, DEV_H]))
        self.kf.x[:4] = convert_bbox_to_z(state)

    def update(self, state: np.ndarray, hist: np.ndarray):
        self.__update_state(state)
        self.__update_hist(hist)

    def __update_state(self, state: np.ndarray):

        pass

    def __update_hist(self, hist: np.ndarray):
        pass

    def predict(self):
        pass

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def get_hist(self):
        return self.hist


if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4])
    tracker = Tracker(arr, arr)
