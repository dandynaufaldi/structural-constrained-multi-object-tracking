import unittest

import numpy as np

from state import DetectionState, ObjectState, StructuralConstraint


def factory_object() -> ObjectState:
    x = 20
    y = 20
    width = 30
    height = 40
    frame_step = 0
    state = ObjectState(x=x, y=y, width=width, height=height, frame_step=frame_step)
    return state


def factory_detection() -> DetectionState:
    x = 30
    y = 30
    width = 30
    height = 40
    frame_step = 1
    state = DetectionState(x=x, y=y, width=width, height=height, frame_step=frame_step)
    return state


class TestDetectionState(unittest.TestCase):

    def test_instantiate(self):
        x = 20
        y = 20
        width = 30
        height = 40
        frame_step = 0
        state = DetectionState(x=x, y=y, width=width, height=height, frame_step=frame_step)

        self.assertIsInstance(state, DetectionState)
        self.assertEqual(state.x, x)
        self.assertEqual(state.y, y)
        self.assertEqual(state.height, height)
        self.assertEqual(state.width, width)
        self.assertEqual(state.frame_step, frame_step)
        self.assertIs(state.histogram, None)

    def test_instantiate_from_bbox(self):
        left = 5
        top = 0
        width = 30
        height = 40
        frame_step = 0
        full_image = np.empty((10, 10, 3))
        state = DetectionState.from_bbox(
            left=left,
            top=top,
            width=width,
            height=height,
            frame_step=frame_step,
            full_image=full_image)

        self.assertIsInstance(state, DetectionState)
        self.assertEqual(state.x, 20)
        self.assertEqual(state.y, 20)
        self.assertIsInstance(state.histogram, np.ndarray)

    def test_instantiate_from_bbox_non_rgb(self):
        left = 5
        top = 0
        width = 30
        height = 40
        frame_step = 0
        full_image = np.empty((10, 10))
        with self.assertRaises(AssertionError):
            state = DetectionState.from_bbox(
                left=left,
                top=top,
                width=width,
                height=height,
                frame_step=frame_step,
                full_image=full_image)


class TestObjectState(unittest.TestCase):

    def test_instantiate(self):
        x = 20
        y = 20
        width = 30
        height = 40
        frame_step = 0
        state = ObjectState(x=x, y=y, width=width, height=height, frame_step=frame_step)

        self.assertIsInstance(state, ObjectState)
        self.assertEqual(state.x, x)
        self.assertEqual(state.y, y)
        self.assertEqual(state.height, height)
        self.assertEqual(state.width, width)
        self.assertEqual(state.frame_step, frame_step)
        self.assertEqual(state.v_x, 0.0)
        self.assertEqual(state.v_y, 0.0)
        self.assertIsNone(state.histogram)

    def test_from_detection(self):
        detection_state = factory_detection()
        object_state = ObjectState.from_detection(detection_state)

        self.assertIsInstance(object_state, ObjectState)
        self.assertEqual(object_state.x, detection_state.x)
        self.assertEqual(object_state.y, detection_state.y)
        self.assertEqual(object_state.height, detection_state.height)
        self.assertEqual(object_state.width, detection_state.width)
        self.assertEqual(object_state.frame_step, detection_state.frame_step)
        self.assertEqual(object_state.v_x, 0.0)
        self.assertEqual(object_state.v_y, 0.0)
        self.assertEqual(object_state.histogram, detection_state.histogram)

    def test_update_from_detection(self):
        detection_state = factory_detection()
        object_state = factory_object()
        delta_time = detection_state.frame_step - object_state.frame_step
        v_x = (detection_state.x - object_state.x) / delta_time
        v_y = (detection_state.y - object_state.y) / delta_time

        object_state.update_from_detection(detection_state)
        self.assertEqual(object_state.x, detection_state.x)
        self.assertEqual(object_state.y, detection_state.y)
        self.assertEqual(object_state.height, detection_state.height)
        self.assertEqual(object_state.width, detection_state.width)
        self.assertEqual(object_state.frame_step, detection_state.frame_step)
        self.assertEqual(object_state.v_x, v_x)
        self.assertEqual(object_state.v_y, v_y)


class TestStructuralConstraint(unittest.TestCase):

    def test_instantiate(self):
        delta_x = 1.5
        delta_y = 2.0
        delta_v_x = 0.5
        delta_v_y = 0.67

        sc = StructuralConstraint(
            delta_x=delta_x, delta_y=delta_y, delta_v_x=delta_v_x, delta_v_y=delta_v_y)
        self.assertIsInstance(sc, StructuralConstraint)
