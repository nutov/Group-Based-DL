import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import base_dataset, TrainDataset, TestDataset


class TestTrainDataset(unittest.TestCase):
    def test_init(self):
        ds = TrainDataset()
        self.assertIsInstance(ds, base_dataset)

class TestTestDataset(unittest.TestCase):
    def test_init(self):
        ds = TestDataset()
        self.assertIsInstance(ds, base_dataset)

if __name__ == '__main__':
    unittest.main()

