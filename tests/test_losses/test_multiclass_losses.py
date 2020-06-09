"""
Module to run unit tests on loss functions and metrics for multi-class problems
"""
import unittest
import numpy as np
import torch
from utils.loss import multi_class_smooth_dice_loss, multi_class_dice_ce_sum

class TestMultiClassMeasure(unittest.TestCase):
    """
    Base class for testing losses. It provides dummy images with varying similarity,
    such that various degrees of the loss functions / metrics can be emulated.
    """
    def setUp(self):
        torch.manual_seed(1001) # manual seed for reproducibility

        # dummy images: batch of 16, four channels
        base = -15 * torch.ones(16, 4, 256, 256)
        # first class is true
        base[:, 0] *= -1

        # any random class except the first class is true
        negative_base = torch.randn(16, 4, 256, 256)
        negative_base[:, 0] = -15 * torch.ones(16, 256, 256)

        # class label target
        self.target_class = torch.argmax(base, axis=1)
        self.negative_target_class = torch.argmax(negative_base, axis=1)
        self.half_negative_target_class = self.target_class.clone()
        self.half_negative_target_class[8:] = self.negative_target_class[8:]

        # assume logits as segmentation mask, predicts all first class
        self.seg_mask_logits = base.clone()

        # move tensors to gpu if available
        if torch.cuda.is_available():
            self.target_class = self.target_class.cuda()
            self.negative_target_class = self.negative_target_class.cuda()
            self.half_negative_target_class = self.half_negative_target_class.cuda()
            self.seg_mask_logits = self.seg_mask_logits.cuda()

    def _test_worstcase(self):
        loss = self.measure(self.seg_mask_logits, self.negative_target_class)
        assert np.isclose(loss.item(), self.worstcase_target, atol=self.worstcase_tol)

    def _test_bestcase(self):
        loss = self.measure(self.seg_mask_logits, self.target_class)
        assert np.isclose(loss.item(), self.bestcase_target, atol=self.bestcase_tol)

    def _test_order(self):
        negative = self.measure(self.seg_mask_logits, self.negative_target_class)
        similar = self.measure(self.seg_mask_logits, self.half_negative_target_class)
        equal = self.measure(self.seg_mask_logits, self.target_class)

        if self.higher_is_better:
            assert negative < similar < equal
        else:
            assert negative > similar > equal

class TestMultiClassSmoothDiceLoss(TestMultiClassMeasure):
    measure = staticmethod(multi_class_smooth_dice_loss)
    bestcase_target = 0.
    bestcase_tol = 1e-6
    worstcase_target = 1.
    worstcase_tol = 1e-6
    higher_is_better = False

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

class TestMultiClassDiceCESumLoss(TestMultiClassMeasure):
    measure = staticmethod(multi_class_dice_ce_sum)
    bestcase_target = 0.
    bestcase_tol = 1e-6
    worstcase_target = 15.5
    worstcase_tol = 1e-6
    higher_is_better = False

    def test_worstcase(self):
        self._test_worstcase()

    def test_bestcase(self):
        self._test_bestcase()

    def test_order(self):
        self._test_order()

if __name__ == "__main__":
    unittest.main()
