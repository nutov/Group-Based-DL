import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import BaseDataset, TrainDataset, TestDataset


class TestTrainDataset(unittest.TestCase):
    def test_init(self):
        ds = TrainDataset()
        self.assertIsInstance(ds, BaseDataset)

class TestTestDataset(unittest.TestCase):
    def test_init(self):
        ds = TestDataset()
        self.assertIsInstance(ds, BaseDataset)

if __name__ == '__main__':
    unittest.main()

