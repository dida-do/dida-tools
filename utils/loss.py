"""
This module contains generic pytorch losses for training.
"""

import sys
import torch
import numpy as np

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

def binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor,
                        weight: float=.99, eps: float=1e-6) -> torch.Tensor:
    
    pred = torch.sigmoid(pred[:,0])
    target = (target[:,0] > 0).float() # TODO here I changed [:,0]
    
    return -1*(weight*target * torch.log(pred+eps) + (1-weight)*(1-target) * torch.log(1-pred+eps)).mean()
    
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

    pred = torch.sigmoid(pred[:,0])
    target = (target[:,0] > 0).float() # TODO here I changed [:,0]
    
    intersection = (pred.view(-1) * target.view(-1)).sum()

    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth + eps))

def smooth_dice_beta_loss(pred: torch.Tensor, target: torch.Tensor,
                     beta: float=1., smooth: float=1., eps: float=1e-6) -> torch.Tensor:
    '''
    Smoothed dice loss.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param smooth: (float) smoothing value
    :param eps: (eps) epsilon for numerical stability

    :returns dice_loss: (torch.Tensor) the dice loss
    '''
    
    pred = torch.sigmoid(pred[:,0])
    target = (target[:,0] > 0).float() # TODO here I changed [:,0]
    
    tp = (pred.view(-1) * target.view(-1)).sum()
    fp = pred.view(-1).sum() - tp
    tn = ((1-pred).view(-1) * (1-target).view(-1)).sum()
    fn = (1-pred).view(-1).sum() - tn
    
    return 1 - (((1+beta)*tp + smooth) / ((1+beta)*tp + beta*fn + fp + smooth + eps))

def inverse_smooth_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float=1., eps: float=1e-6) -> torch.Tensor:
    '''
    Smoothed dice loss.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param smooth: (float) smoothing value
    :param eps: (eps) epsilon for numerical stability

    :returns dice_loss: (torch.Tensor) the dice loss
    '''
    
    pred = 1-torch.sigmoid(pred[:,0])
    target = 1-(target[:,0] > 0).float() # TODO here I changed [:,0]
    
    intersection = (pred.view(-1) * target.view(-1)).sum()

    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth + eps))

def precision(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the precision.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns precision: (torch.Tensor)
    '''
    #print("call will precision on {} / {}\n".format(pred.shape, target.shape))
    pred = (pred[:,0] > 0).float()
    target = (target[:,0] > 0).float()
    
    tp = ((pred == 1.) * (target == 1.)).float() # TODO why not (pred * target) ?

    return tp.sum() / (pred.sum() + eps)

def recall(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the recall.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns recall: (torch.Tensor)
    '''
    #print("call will recall on {} / {}\n".format(pred.shape, target.shape))
    pred = (pred[:,0] > 0).float()
    target = (target[:,0] > 0).float()

    tp = ((pred == 1.) * (target == 1.)).float()

    return tp.sum() / (target.sum() + eps)

def f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Function to calculate the f1 score.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target

    :returns f1: (torch.Tensor)
    '''
    #print("call will f1 on {} / {}\n".format(pred.shape, target.shape))
    p = precision(pred, target)
    r = recall(pred, target)
    #print(p, r)

    return (2 * p * r) / (p + r)

def masked_smooth_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float=1., eps: float=1e-6) -> torch.Tensor:
    '''
    Smoothed dice loss.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param smooth: (float) smoothing value
    :param eps: (eps) epsilon for numerical stability

    :returns dice_loss: (torch.Tensor) the dice loss
    '''
    
    mask = target[:,1].float()
    pred = torch.sigmoid(pred)
    target = (target[:,0] > 0).float()
    
    pred = mask*pred[:,0]
    target = mask*target

    intersection = (pred.view(-1) * target.view(-1)).sum()

    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth + eps))

def masked_precision(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the precision.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns precision: (torch.Tensor)
    '''
    
    mask = target[:,1].float()
    pred = mask*(pred > 0).float()
    target = mask*(target[:,0] > 0).float()

    tp = ((pred == 1) * (target == 1)).float()

    return tp.sum() / (pred.sum() + eps)

def masked_recall(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    '''
    Function to calculate the recall.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target
    :param eps: (eps) epsilon for numerical stability

    :returns recall: (torch.Tensor)
    '''
    
    mask = target[:,1].float()
    pred = mask*(pred > 0).float()
    target = mask*(target[:,0] > 0).float()

    tp = ((pred == 1) * (target == 1)).float()

    return tp.sum() / (target.sum() + eps)

def masked_f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Function to calculate the f1 score.

    :param pred: (torch.Tensor) predictions
    :param target: (torch.Tensor) target

    :returns f1: (torch.Tensor)
    '''
    p = masked_precision(pred, target)
    r = masked_recall(pred, target)

    return (2 * p * r) / (p + r)
