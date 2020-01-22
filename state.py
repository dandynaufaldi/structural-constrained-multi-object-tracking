from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionState:
    x: float
    y: float
    width: float
    height: float
    frame_step: int
    histogram: Optional[np.ndarray] = None


@dataclass
class ObjectState:
    x: float
    y: float
    width: float
    height: float
    frame_step: int
    v_x: float = 0.0
    v_y: float = 0.0
    histogram: Optional[np.ndarray] = None

    @staticmethod
    def from_detection(detection: DetectionState) -> 'ObjectState':
        return ObjectState(**asdict(detection))

    def update_from_detection(self, detection: DetectionState):
        delta_time = detection.frame_step - self.frame_step
        self.v_x = (detection.x - self.x) / delta_time
        self.v_y = (detection.y - self.y) / delta_time
        self.x = detection.x
        self.y = detection.y
        self.width = detection.width
        self.height = detection.height
        self.frame_step = detection.frame_step
