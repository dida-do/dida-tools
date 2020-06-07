"""
Module to run unit tests on loss functions and metrics when usability mask is available
"""
import unittest
import numpy as np
import torch
from utils.loss import masked_precision, masked_recall, masked_f1, masked_smooth_dice_loss

class TestMeasure(unittest.TestCase):
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

    def _test_worstcase(self):
        loss = self.measure(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        assert np.isclose(loss.item(), self.worstcase_target, atol=self.worstcase_tol)

    def _test_bestcase(self):
        loss = self.measure(self.seg_mask_logits[:, 1:], self.seg_mask_logits)
        assert np.isclose(loss.item(), self.bestcase_target, atol=self.bestcase_tol)

    def _test_order(self):
        negative = self.measure(self.seg_mask_logits[:, 1:], self.negative_seg_mask_logits)
        similar = self.measure(self.seg_mask_logits[:, 1:], self.half_negative_logits)
        equal = self.measure(self.seg_mask_logits[:, 1:], self.seg_mask_logits)

        if self.higher_is_better:
            assert negative < similar < equal
        else:
            assert negative > similar > equal

class TestMaskedPrecision(TestMeasure):
    measure = staticmethod(masked_precision)
    bestcase_target = 1.
    bestcase_tol = 1e-6
    worstcase_target = 0.
    worstcase_tol = 1e-6
    higher_is_better = True

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

class TestMaskedRecall(TestMeasure):
    measure = staticmethod(masked_recall)
    bestcase_target = 1.
    bestcase_tol = 1e-6
    worstcase_target = 0.
    worstcase_tol = 1e-6
    higher_is_better = True

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

class TestMaskedF1Score(TestMeasure):
    measure = staticmethod(masked_f1)
    bestcase_target = 1.
    bestcase_tol = 1e-6
    worstcase_target = 0.
    worstcase_tol = 1e-6
    higher_is_better = True

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

class TestSmoothDiceLoss(TestMeasure):
    measure = staticmethod(masked_smooth_dice_loss)
    bestcase_target = 0.
    bestcase_tol = 1e-2
    worstcase_target = 1.
    worstcase_tol = 1e-2
    higher_is_better = False

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

if __name__ == "__main__":
    unittest.main()
