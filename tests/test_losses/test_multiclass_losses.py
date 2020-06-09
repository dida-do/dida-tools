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
        base = torch.randn(16, 4, 256, 256)

        # class label target
        self.target_class = torch.argmax(base, axis=1)
        self.negative_target_class = torch.argmax(-1. * base, axis=1)

        # one-hot target
        self.target_one_hot = torch.nn.functional.one_hot(self.target_class, num_classes=4)
        self.target_one_hot = self.target_one_hot.permute(0, 3, 1, 2)

        self.negative_target_one_hot = torch.nn.functional.one_hot(self.negative_target_class, num_classes=4)
        self.negative_target_one_hot = self.negative_target_one_hot.permute(0, 3, 1, 2)

        # assume logits as segmentation mask
        self.seg_mask_logits = 10.*base - 5.
        self.negative_seg_mask_logits = -1*self.seg_mask_logits
        self.half_negative_logits = self.seg_mask_logits.clone()
        self.half_negative_logits[:, :2] = self.negative_seg_mask_logits[:, :2]

        # move tensors to gpu if available
        if torch.cuda.is_available():
            self.target_class = self.target_class.cuda()
            self.negative_target_class = self.negative_target_class.cuda()
            self.target_one_hot = self.target_one_hot.cuda()
            self.negative_target_one_hot = self.negative_target_one_hot.cuda()
            self.seg_mask_logits = self.seg_mask_logits.cuda()
            self.negative_seg_mask_logits = self.negative_seg_mask_logits.cuda()
            self.half_negative_logits = self.half_negative_logits.cuda()

    def _test_worstcase(self):
        loss = self.measure(self.seg_mask_logits, self.negative_target_class)
        assert np.isclose(loss.item(), self.worstcase_target, atol=self.worstcase_tol)

    def _test_bestcase(self):
        loss = self.measure(self.seg_mask_logits, self.target_class)
        assert np.isclose(loss.item(), self.bestcase_target, atol=self.bestcase_tol)

    def _test_order(self):
        negative = self.measure(self.negative_seg_mask_logits, self.target_class)
        similar = self.measure(self.half_negative_logits, self.target_class)
        equal = self.measure(self.seg_mask_logits, self.target_class)

        if self.higher_is_better:
            assert negative < similar < equal
        else:
            assert negative > similar > equal

class TestMultiClassSmoothDiceLoss(TestMultiClassMeasure):
    measure = staticmethod(multi_class_smooth_dice_loss)
    bestcase_target = 0.
    bestcase_tol = 1e26
    worstcase_target = 1.
    worstcase_tol = 1e-2
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
