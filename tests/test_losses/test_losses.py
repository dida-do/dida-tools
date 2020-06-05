"""
Module to run unit tests on loss functions and metrics
"""
import unittest
import numpy as np
import torch
from utils.loss import precision, recall, f1, smooth_dice_loss, smooth_dice_beta_loss, dice_bce_sum

class TestLoss(unittest.TestCase):
    """
    Base class for testing losses. It provides dummy images with varying similarity,
    such that various degrees of the loss functions / metrics can be emulated.
    """
    def setUp(self):
        torch.manual_seed(1001) # manual seed for reproducibility

        # dummy images: batch of 16, four channels
        base = (torch.randn(16, 4, 256, 256) > 0.).float()

        # assume logits as segmentation mask
        self.seg_mask_logits = 10. * base - 5.
        self.negative_seg_mask_logits = -1 * self.seg_mask_logits
        self.half_negative_logits = self.seg_mask_logits.clone()
        self.half_negative_logits[:, :2] = self.negative_seg_mask_logits[:, :2]

        # move tensors to gpu if available
        if torch.cuda.is_available():
            self.seg_mask_logits = self.seg_mask_logits.cuda()
            self.negative_seg_mask_logits = self.negative_seg_mask_logits.cuda()
            self.half_negative_logits = self.half_negative_logits.cuda()

class TestPrecision(TestLoss):
    def test_worstcase(self):
        loss = precision(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = precision(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = precision(self.seg_mask_logits, self.negative_seg_mask_logits)
        medium_loss = precision(self.seg_mask_logits, self.half_negative_logits)
        large_loss = precision(self.seg_mask_logits, self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestRecall(TestLoss):
    def test_worstcase(self):
        loss = recall(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = recall(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = recall(self.seg_mask_logits, self.negative_seg_mask_logits)
        medium_loss = recall(self.seg_mask_logits, self.half_negative_logits)
        large_loss = recall(self.seg_mask_logits, self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestF1Score(TestLoss):
    def test_worstcase(self):
        loss = f1(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = f1(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = f1(self.seg_mask_logits, self.negative_seg_mask_logits)
        medium_loss = f1(self.seg_mask_logits, self.half_negative_logits)
        large_loss = f1(self.seg_mask_logits, self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestSmoothDiceLoss(TestLoss):
    def test_worstcase(self):
        loss = smooth_dice_loss(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-2)

    def test_bestcase(self):
        loss = smooth_dice_loss(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 0., atol=1e-2)

    def test_order(self):
        small_loss = smooth_dice_loss(self.seg_mask_logits, self.seg_mask_logits)
        medium_loss = smooth_dice_loss(self.seg_mask_logits, self.half_negative_logits)
        large_loss = smooth_dice_loss(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestSmoothDiceBetaLoss(TestLoss):
    def test_worstcase(self):
        loss = smooth_dice_beta_loss(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-2)

    def test_bestcase(self):
        loss = smooth_dice_beta_loss(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 0., atol=1e-2)

    def test_order(self):
        small_loss = smooth_dice_beta_loss(self.seg_mask_logits, self.seg_mask_logits)
        medium_loss = smooth_dice_beta_loss(self.seg_mask_logits, self.half_negative_logits)
        large_loss = smooth_dice_beta_loss(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestDiceBCESumLoss(TestLoss):
    def test_worstcase(self):
        loss = dice_bce_sum(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert np.isclose(loss.item(), 3., atol=1e-2)

    def test_bestcase(self):
        loss = dice_bce_sum(self.seg_mask_logits, self.seg_mask_logits)
        assert np.isclose(loss.item(), 0., atol=1e-2)

    def test_order(self):
        small_loss = dice_bce_sum(self.seg_mask_logits, self.seg_mask_logits)
        medium_loss = dice_bce_sum(self.seg_mask_logits, self.half_negative_logits)
        large_loss = dice_bce_sum(self.seg_mask_logits, self.negative_seg_mask_logits)
        assert large_loss > medium_loss > small_loss

if __name__ == "__main__":
    unittest.main()
