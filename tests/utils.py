import numpy as np

from sc_tracker.state import DetectionState, ObjectState


def factory_object() -> ObjectState:
    x = 20
    y = 20
    width = 30
    height = 40
    frame_step = 0
    state = ObjectState(
        x=x, y=y, width=width, height=height, frame_step=frame_step, histogram=np.empty(8)
    )
    return state


def factory_detection() -> DetectionState:
    x = 30
    y = 30
    width = 30
    height = 40
    frame_step = 1
    state = DetectionState(
        x=x, y=y, width=width, height=height, frame_step=frame_step, histogram=np.empty(8)
    )
    return state
