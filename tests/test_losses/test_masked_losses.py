"""
Module to run unit tests on loss functions and metrics when usability mask is available
"""
import unittest
import numpy as np
import torch
from utils.loss import masked_precision, masked_recall, masked_f1, masked_smooth_dice_loss

class TestMaskedLoss(unittest.TestCase):
    """
    Base class for testing losses that consider the usability mask. It provides dummy
    images with varying similarity, such that various degrees of the loss functions / metrics
    can be emulated. The first channel of the images will contain a usability mask.
    """
    def setUp(self):
        torch.manual_seed(1001) # manual seed for reproducibility

        # dummy images: batch of 16, four channels + 1 usability mask
        base = (torch.randn(16, 4, 256, 256) > 0.).float()

        # assume logits as segmentation mask
        self.seg_mask_logits = torch.ones(16, 5, 256, 256)
        self.seg_mask_logits[:, 1:] = 10.*base - 5.
        self.negative_seg_mask_logits = -1*self.seg_mask_logits
        self.half_negative_logits = self.seg_mask_logits.clone()
        self.half_negative_logits[:, :2] = self.negative_seg_mask_logits[:, :2]

        # expand by usability mask
        usability_mask = (torch.randn(16, 256, 256) > 0.).float()
        self.seg_mask_logits[:, 0] = usability_mask
        self.negative_seg_mask_logits[:, 0] = usability_mask
        self.half_negative_logits[:, 0] = usability_mask

        # move tensors to gpu if available
        if torch.cuda.is_available():
            self.seg_mask_logits = self.seg_mask_logits.cuda()
            self.negative_seg_mask_logits = self.negative_seg_mask_logits.cuda()
            self.half_negative_logits = self.half_negative_logits.cuda()

class TestMaskedPrecision(TestMaskedLoss):
    def test_worstcase(self):
        loss = masked_precision(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = masked_precision(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = masked_precision(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        medium_loss = masked_precision(self.seg_mask_logits[:, 1:], self.half_negative_logits)
        large_loss = masked_precision(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestMaskedRecall(TestMaskedLoss):
    def test_worstcase(self):
        loss = masked_recall(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = masked_recall(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = masked_recall(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        medium_loss = masked_recall(self.seg_mask_logits[:, 1:], self.half_negative_logits)
        large_loss = masked_recall(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestMaskedF1Score(TestMaskedLoss):
    def test_worstcase(self):
        loss = masked_f1(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert loss == 0.

    def test_bestcase(self):
        loss = masked_f1(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-6)

    def test_order(self):
        small_loss = masked_f1(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        medium_loss = masked_f1(self.seg_mask_logits[:, 1:], self.half_negative_logits)
        large_loss = masked_f1(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert large_loss > medium_loss > small_loss

class TestSmoothDiceLoss(TestMaskedLoss):
    def test_worstcase(self):
        loss = masked_smooth_dice_loss(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert np.isclose(loss.item(), 1., atol=1e-2)

    def test_bestcase(self):
        loss = masked_smooth_dice_loss(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert np.isclose(loss.item(), 0., atol=1e-2)

    def test_order(self):
        small_loss = masked_smooth_dice_loss(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        medium_loss = masked_smooth_dice_loss(self.seg_mask_logits[:, 1:], self.half_negative_logits)
        large_loss = masked_smooth_dice_loss(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert large_loss > medium_loss > small_loss

if __name__ == "__main__":
    unittest.main()
