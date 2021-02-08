"""Tests for the utilities in `mutil_channel_conv_nets`"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
import unittest

from models.multi_channel_conv_nets import modify_conv, P2PModel, binary_clf_model
from utils.torchutils import forward, backprop, assert_not_nan, assert_isfinite

class TestModifyConv(unittest.TestCase):
    def test_modify_conv(self):
        layer = nn.Conv2d(3, 6, (3, 3))
        new_layer = modify_conv(layer, 4)
        
        assert new_layer.ch_in == 4
        
class TestP2PModel(unittest.TestCase):
    def setUp(self):
        self.model = P2PModel()
        
    def test_parameters(self):
        for tensor in self.model.parameters():
            assert_not_nan(tensor)
            assert_isfinite(tensor)
            
    def test_forward(self):
        test_array = torch.randn(1, 3, 256, 256)
        pred = forward(self.model, test_array, self.device)
        assert tuple(pred.shape) == (1, 3, 256, 256)
        assert_isfinite(pred)
        assert_not_nan(pred)
        
class TestBinaryClf(unittest.Testcase):
    def setUp(self):
        self.model = binary_clf_model()
        
    def test_parameters(self):
        for tensor in self.model.parameters():
            assert_not_nan(tensor)
            assert_isfinite(tensor)
            
    def test_forward(self):
        test_array = torch.randn(1, 3, 256, 256)
        pred = forward(self.model, test_array, self.device)
        assert tuple(pred.shape) == (1, )
        assert_isfinite(pred)
        assert_not_nan(pred)
        
if __name__ == "__main__":
    unittest.main()