import sys 
sys.path.append("./../../")
import unittest
from unittest import TestCase
import numpy as np
from vinecopulaslab.partnerselection.geometric import GeometricSelection

class TestDownload(TestCase):
    def test_distance_to_diagonal(self):
        # test template
        # TODO: add tests
        line = np.array([1, 1, 1])
        pts = np.array([[0, 0, 0]])
        self.assertEqual(GeometricSelection.distance_to_line(line, pts), 0, "Should be 0") 


if __name__ == '__main__':
    unittest.main()