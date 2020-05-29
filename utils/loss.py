"""
This module contains generic pytorch losses for training.
"""

import sys
import torch
from torch.nn import functional as F

def get_loss(name: str):
    """
    Load a loss by name from torch.nn.modules.loss
    and revert to this module as fallback routine.
    Note: importing the losses by name is case sensitive.
    Torch losses are instantiated with default arguments
    and then returned. Losses in this module
    are directly returned as functions.
    """
    from torch.nn.modules import loss as torchloss
    utilsloss = sys.modules[__name__]
    try:
        loss = getattr(torchloss, name)
        loss = loss()
    except AttributeError:
        loss = getattr(utilsloss, name)
    del torchloss, utilsloss
    return loss

def smooth_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float=1., eps: float=1e-6) -> torch.Tensor:
    '''
    Smoothed dice loss.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param smooth: (float) smoothing value
    :param eps: (eps) epsilon for numerical stability

    :returns dice_loss: (torch.Tensor) the dice loss
    '''
    pred = torch.sigmoid(pred)
    target = (target > 0).float()

    intersection = (pred.view(-1) * target.view(-1)).sum()

    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth + eps))

def dice_bce_sum(pred: torch.Tensor,
                 target: torch.Tensor,
                 weight: float=0.5,
                 smooth: float=1.,
                 eps: float=1e-6):
    """Weighted sum of smooth dice loss and binary cross entropy.
    
    `weight` is the relative weight between the two loss functions."""
    
    return weight * smooth_dice_loss(pred, target, smooth, eps) + (1 - weight) * F.binary_cross_entropy_with_logits(pred, target)

def multi_class_smooth_dice_loss(pred: torch.Tensor,
                                 target: torch.Tensor,
                                 smooth: float=1.,
                                 eps: float=1e-6):
    """Smooth Dice loss for multi class classification."""
    
    prob = F.softmax(pred, dim=1)
    
    with torch.no_grad():
        num_class = pred.size(1)
        targets_oh = torch.eye(num_class, device=pred.get_device())[targets.squeeze(1)]
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()
        
        dims = (0,) + tuple(range(2, targets_oh.ndimension()))
        
    intersect = torch.sum(prob * targets_oh, dims)
    cardinality = torch.sum(prob + targets_oh, dims)
        
    return 1. - ((2. * intersect + self.smooth) / (cardinality + self.smooth)).mean()
    
def multi_class_dice_ce_sum(pred: torch.Tensor,
                            target: torch.Tensor,
                            weight: float=0.5,
                            smooth: float=1.,
                            eps: float=1e-6):
    """Sum of cross entropy and dice loss for multi class case"""
    
    target = target.long()
    
    return weight * multi_class_smooth_dice_loss(pred, target, smooth, eps) + (1. - weight) * F.cross_entropy(pred, target)

def precision(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the precision.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns precision: (torch.Tensor)
    '''
    pred = (pred > 0).float()
    target = (target > 0).float()

    tp = ((pred == 1) * (target == 1)).float()

    return tp.sum() / (pred.sum() + eps)

def recall(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the recall.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns recall: (torch.Tensor)
    '''
    pred = (pred > 0).float()
    target = (target > 0).float()

    tp = ((pred == 1) * (target == 1)).float()

    return tp.sum() / (target.sum() + eps)

def f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Function to calculate the f1 score.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target

    :returns f1: (torch.Tensor)
    '''
    p = precision(pred, target)
    r = recall(pred, target)

    return (2 * p * r) / (p + r)
