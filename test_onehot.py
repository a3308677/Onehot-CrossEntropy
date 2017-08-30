import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np

from unittest import TestCase
from onehot import onehot, test1d


class TestOnehot(TestCase):
    def test_onehot_1d(self):
        for _ in range(100):
            test1d()

    def test_function(self):
        self.assertEqual(True, True)
