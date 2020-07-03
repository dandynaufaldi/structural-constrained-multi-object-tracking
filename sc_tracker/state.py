from typing import List, Tuple

import numpy as np
from dataclasses import asdict, dataclass, field

ObjectStateData = Tuple[float, float, float, float, float, float]
DetectionStateData = Tuple[float, float, float, float]
StructuralConstraintData = Tuple[float, float, float, float]
HistogramData = List[float]


class Index:
    INDEX_X = 0
    INDEX_Y = 1
    INDEX_W = 2
    INDEX_H = 3
    INDEX_VX = 4
    INDEX_VY = 5
    INDEX_DX = 0
    INDEX_DY = 1
    INDEX_DVX = 2
    INDEX_DVY = 3


@dataclass
class DetectionState:
    x: float
    y: float
    width: float
    height: float
    frame_step: int
    histogram: np.ndarray = field(repr=False)
    internal_state: DetectionStateData = field(init=False)

    def __post_init__(self):
        self.__set_state()

    @staticmethod
    def from_bbox(
        left: int,
        top: int,
        width: int,
        height: int,
        frame_step: int,
        full_image: np.ndarray,
        n_bins: int = 8,
    ) -> "DetectionState":
        left = int(max(left, 0))
        top = int(max(top, 0))
        x = left + width / 2
        y = top + height / 2
        histogram = None
        if full_image is not None:
            shape = full_image.shape
            assert len(shape) == 3 and shape[2] == 3, (
                "full_image should be in RGB/BGR with shape (height, width, 3) "
                f"but found {shape}"
            )
            image = full_image[top : top + int(height), left : left + int(width)]
            graysacle = image.mean(axis=2)
            bins = np.arange(n_bins + 1) * (256 // n_bins)
            histogram, _ = np.histogram(graysacle, bins=bins)
            histogram = histogram / histogram.sum()
        return DetectionState(
            x=x, y=y, width=width, height=height, frame_step=frame_step, histogram=histogram,
        )

    def __set_state(self):
        self.internal_state = (self.x, self.y, self.width, self.height)

    def state(self) -> DetectionStateData:
        return self.internal_state


@dataclass
class ObjectState:
    x: float
    y: float
    width: float
    height: float
    frame_step: int
    histogram: np.ndarray = field(repr=False)
    v_x: float = 0.0
    v_y: float = 0.0
    internal_state: ObjectStateData = field(init=False)

    def __post_init__(self):
        self.__set_state()

    @staticmethod
    def from_detection(detection: DetectionState) -> "ObjectState":
        attrs = asdict(detection)
        attrs.pop("internal_state", None)
        return ObjectState(**attrs)

    def update_from_detection(self, detection: DetectionState):
        delta_time = detection.frame_step - self.frame_step
        self.v_x = (detection.x - self.x) / delta_time
        self.v_y = (detection.y - self.y) / delta_time
        self.x = detection.x
        self.y = detection.y
        self.width = detection.width
        self.height = detection.height
        self.frame_step = detection.frame_step
        self.histogram = detection.histogram.copy()
        self.__set_state()

    def update_from_object_state(self, object_state: "ObjectState"):
        self.x = object_state.x
        self.y = object_state.y
        self.v_x = object_state.v_x
        self.v_y = object_state.v_y
        self.width = object_state.width
        self.height = object_state.width
        self.frame_step = object_state.frame_step
        self.histogram = object_state.histogram.copy()
        self.__set_state()

    def __set_state(self):
        self.internal_state = (self.x, self.y, self.width, self.height, self.v_x, self.v_y)

    def state(self) -> ObjectStateData:
        return self.internal_state


@dataclass
class StructuralConstraint:
    delta_x: float
    delta_y: float
    delta_v_x: float
    delta_v_y: float
    internal_state: StructuralConstraintData = field(init=False)

    def __init__(self, delta_x: float, delta_y: float, delta_v_x: float, delta_v_y: float):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_v_x = delta_v_x
        self.delta_v_y = delta_v_y
        self.__set_state()

    @staticmethod
    def create(first_object: ObjectState, second_object: ObjectState) -> "StructuralConstraint":
        delta_x = first_object.x - second_object.x
        delta_y = first_object.y - second_object.y
        delta_v_x = first_object.v_x - second_object.v_x
        delta_v_y = first_object.v_y - second_object.v_y

        return StructuralConstraint(
            delta_x=delta_x, delta_y=delta_y, delta_v_x=delta_v_x, delta_v_y=delta_v_y
        )

    def update_from_sc(self, sc: "StructuralConstraint"):
        self.delta_x = sc.delta_x
        self.delta_y = sc.delta_y
        self.delta_v_x = sc.delta_v_x
        self.delta_v_y = sc.delta_v_y
        self.__set_state()

    def __set_state(self):
        self.internal_state = (self.delta_x, self.delta_y, self.delta_v_x, self.delta_v_y)

    def state(self) -> StructuralConstraintData:
        return self.internal_state
