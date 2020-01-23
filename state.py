from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class DetectionState:
    x: float
    y: float
    width: float
    height: float
    frame_step: int
    histogram: Optional[np.ndarray] = None

    @staticmethod
    def from_bbox(left: int,
                  top: int,
                  width: int,
                  height: int,
                  frame_step: int,
                  full_image: Optional[np.ndarray] = None,
                  bins: Optional[Union[list, np.ndarray]] = np.arange(9) * 32) -> 'DetectionState':
        x = left + width / 2
        y = top + height / 2
        histogram = None
        if full_image is not None:
            # RGB or BGR image
            assert len(full_image.shape) == 3
            image = full_image[top:top + height, left:left + width]
            graysacle = image.mean(axis=2)
            histogram, _ = np.histogram(graysacle, bins=bins)
        return DetectionState(
            x=x, y=y, width=width, height=height, frame_step=frame_step, histogram=histogram)


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


@dataclass
class StructuralConstraint:
    delta_x: float
    delta_y: float
    delta_v_x: float
    delta_v_y: float
