import unittest

import numpy as np

from sc_tracker.cost import calculate_fs, calculate_structural_constraint
from sc_tracker.state import StructuralConstraint
from utils import factory_detection, factory_object


class TestCalculateStructuralConstraint(unittest.TestCase):
    def test_success(self):
        n_object = 4
        objects = [factory_object() for _ in range(n_object)]
        structural_constraints = calculate_structural_constraint(objects)
        self.assertTrue(len(structural_constraints) == n_object)
        self.assertTrue(len(structural_constraints[0]) == n_object)
        for i in range(n_object):
            for j in range(n_object):
                sc = structural_constraints[i][j]
                if i == j:
                    self.assertIsNone(sc)
                else:
                    self.assertIsInstance(sc, StructuralConstraint)


class TestCalculateFs(unittest.TestCase):
    def test_success(self):
        n_object = 4
        n_detection = 3
        objects = [factory_object() for _ in range(n_object)]
        detections = [factory_detection() for _ in range(n_detection)]
        fs = calculate_fs(objects, detections)
        self.assertIsInstance(fs, np.ndarray)
        self.assertEqual(fs.shape, (n_object, n_detection))
