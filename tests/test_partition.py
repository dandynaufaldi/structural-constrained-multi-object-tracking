import math
import unittest

import numpy as np

from partition import gating, possible_assignment_generator, subgroup_by_cluster
from utils import factory_detection, factory_object


class TestGating(unittest.TestCase):
    def test_gating_mismatch_dimension(self):
        n_object = 4
        n_detection = 3

        objects = [factory_object() for _ in range(n_object)]
        detections = [factory_detection() for _ in range(n_detection)]
        fs_matrix = np.empty((n_object, n_detection - 1))
        with self.assertRaises(AssertionError):
            _ = gating(fs_matrix, objects, detections)

    def test_gating_success(self):
        n_object = 4
        n_detection = 3

        objects = [factory_object() for _ in range(n_object)]
        detections = [factory_detection() for _ in range(n_detection)]
        fs_matrix = np.empty((n_object, n_detection))

        mask = gating(fs_matrix, objects, detections)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (n_object, n_detection))


class TestSubGroup(unittest.TestCase):
    def test_by_cluster(self):
        n_object = 20
        n_member = 5
        objects = [factory_object() for _ in range(n_object)]
        for i in range(len(objects)):
            objects[i].x += i
            objects[i].y += i
        subgroup = subgroup_by_cluster(objects, n_member=n_member)
        self.assertEqual(len(subgroup), math.ceil(n_object / n_member))


class TestAssignmentGenerator(unittest.TestCase):
    def test_generator_success(self):
        n_object = 6
        n_detection = 4
        assignment_matrix = np.ones((n_object, n_detection), dtype="int")
        subgroups = np.arange(n_object)
        np.random.shuffle(subgroups)
        subgroups = [subgroups[: n_object // 2], subgroups[n_object // 2 :]]
        for subgroup in subgroups:
            for assignment in possible_assignment_generator(assignment_matrix, subgroup):
                self.assertEqual(len(assignment), len(subgroup))
