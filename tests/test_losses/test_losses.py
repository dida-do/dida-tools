import unittest
import numpy as np
import torch
from utils.loss import precision, recall, f1, smooth_dice_loss, smooth_dice_beta_loss

class TestLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1001) # manual seed for reproducibility

        # dummy images: batch of 16, four channels
        self.all_ones = torch.ones(16, 4, 256, 256)
        self.all_zeros = torch.zeros(16, 4, 256, 256)
        self.mixed = torch.ones(16, 4, 256, 256)
        self.mixed[:, :2] = torch.zeros(16, 2, 256, 256)
        idx = torch.randperm(self.mixed.nelement())
        self.mixed = self.mixed.view(-1)[idx].view(self.mixed.size())
        self.inverted_mixed = 1 - self.mixed
        self.half = self.mixed.clone()
        self.half[:, :2] = self.inverted_mixed[:, :2]

class TestPrecision(TestLoss):
    def test_worstcase(self):
        loss = precision(self.mixed, self.inverted_mixed)
        assert loss == 0.

    def test_bestcase(self):
        loss = precision(self.mixed, self.mixed)
        assert np.isclose(loss, 1., atol=1e-6)

    def test_order(self):
        small_loss = precision(self.mixed, self.inverted_mixed)
        medium_loss = precision(self.mixed, self.half)
        large_loss = precision(self.mixed, self.mixed)
        assert large_loss > medium_loss > small_loss

class TestRecall(TestLoss):
    def test_worstcase(self):
        loss = recall(self.mixed, self.inverted_mixed)
        assert loss == 0.

    def test_bestcase(self):
        loss = recall(self.mixed, self.mixed)
        assert np.isclose(loss, 1., atol=1e-6)

    def test_order(self):
        small_loss = recall(self.mixed, self.inverted_mixed)
        medium_loss = recall(self.mixed, self.half)
        large_loss = recall(self.mixed, self.mixed)
        assert large_loss > medium_loss > small_loss

class TestF1Score(TestLoss):
    def test_worstcase(self):
        loss = f1(self.mixed, self.inverted_mixed)
        assert loss == 0.

    def test_bestcase(self):
        loss = f1(self.mixed, self.mixed)
        assert np.isclose(loss, 1., atol=1e-6)

    def test_order(self):
        small_loss = f1(self.mixed, self.inverted_mixed)
        medium_loss = f1(self.mixed, self.all_ones)
        large_loss = f1(self.mixed, self.mixed)
        assert large_loss > medium_loss > small_loss

class TestSmoothDiceLoss(TestLoss):
    def test_worstcase(self):
        assert True #TODO

    def test_bestcase(self):
        assert True #TODO

    def test_order(self):
        small_loss = smooth_dice_loss(self.mixed, self.mixed)
        medium_loss = smooth_dice_loss(self.mixed, self.half)
        large_loss = smooth_dice_loss(self.mixed, self.inverted_mixed)
        assert large_loss > medium_loss > small_loss

class TestSmoothDiceBetaLoss(TestLoss):
    def test_worstcase(self):
        assert True #TODO

    def test_bestcase(self):
        assert True #TODO

    def test_order(self):
        small_loss = smooth_dice_beta_loss(self.mixed, self.mixed)
        medium_loss = smooth_dice_beta_loss(self.mixed, self.half)
        large_loss = smooth_dice_beta_loss(self.mixed, self.inverted_mixed)
        assert large_loss > medium_loss > small_loss

if __name__ == "__main__":
    unittest.main()
