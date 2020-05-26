"""Main pytorch lightning module for binary segmentation, also supports multiple sigmoid predictions per pixel"""

from training.lightning_modules.base_lightning_module import BaseModule
from utils.data.datasets import SegmentationDataset
from utils.preprocessing import get_preprocess, augmentation
from utils.loss import smooth_dice_loss, dice_bce_sum
from utils.objectise import objectise
from utils.torchutils import visualise

import os
import torch
from torch import nn
import torchvision
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from sklearn.metrics import f1_score

LOSSES = {
    "BCE": nn.BCEWithLogitsLoss,
    "dice": objectise(smooth_dice_loss),
    "dice_bce_sum": objectise(dice_bce_sum)
}

class BinarySegmentationModule(BaseModule):
    """Lightning module for binary semantic segmentation"""
    
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.loss = LOSSES[hparams.loss]()
            
        if self.adv_model is not None:
            raise NotImplementedError("Adversarial loss cannot be used for segmentation.")
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        self.unfreeze_layers()
        logdir = {"train_loss": loss}
        
        return {"loss": loss, "progress_bar": logdir, "log": logdir}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        loss = self.loss(y_hat, y)
        
        y_hat = y_hat > 0.
        
        tp = ((y == 1) & (y_hat == 1)).sum()
        fp = ((y == 1) & (y_hat == 0)).sum()
        tn = ((y == 0) & (y_hat == 0)).sum()
        fn = ((y == 0) & (y_hat == 1)).sum()
        
        return {"val_loss": loss, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    
    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tp = torch.stack([x['tp'] for x in outputs]).sum()
        fp = torch.stack([x['fp'] for x in outputs]).sum()
        tn = torch.stack([x['tn'] for x in outputs]).sum()
        fn = torch.stack([x['fn'] for x in outputs]).sum()
        
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-5)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        
        out_dict = {"val_loss": avg_val_loss, "val_acc": accuracy, "val_f1": f1, "val_precision": precision, "val_recall": recall}
        return {"log": out_dict, "val_loss": avg_val_loss}
    
    def on_epoch_end(self):
        """Write images to tensorboard for inspection"""
        
        if self.inspection_batch is None:
            x, y = next(iter(self._lazy_val_dataloader[0]))
            x = x[:min(32, len(x))]
            y = y[:min(32, len(x))]
        
            grid = torchvision.utils.make_grid(visualise(x))
            self.logger.experiment.add_image(f'initial_imgs', grid, 0)
            
            for ch in range(y.shape[1]):
                grid = torchvision.utils.make_grid(y[:, [ch]])
                self.logger.experiment.add_image(f'ground_truth_channel_{ch}', grid, 0)
            
            self.inspection_batch = x
        
        device = next(self.model.parameters()).device
        
        model_output = self.model(self.inspection_batch.to(device))
            
        batch = torch.sigmoid(model_output.cpu().float())
        for ch in range(batch.shape[1]):
            grid = torchvision.utils.make_grid(batch[:, [ch]])
            self.logger.experiment.add_image(f'predicted_output_channel_{ch}', grid, self.current_epoch)
    
    def __dataloader(self, tfms, fnames):
        batch_size = self.hparams.batch_size
        preprocess = get_preprocess(n_ch=self.hparams.channels)
        
        if hasattr(self.hparams, "min_val"):
            min_val = self.hparams.min_val
        else:
            min_val = 0
        
        dataset = SegmentationDataset(self.data_root,
                                      tfms, preprocess,
                                      fnames=fnames,
                                      min_val=min_val,
                                      max_val=self.hparams.max_val)
        
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=self.hparams.num_workers)
    
    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.no_data_aug:
            tfms = None
        else:
            tfms = augmentation(0.9)
            
        return self.__dataloader(tfms, self.train_fnames)
    
    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader(None, self.val_fnames)
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = self.__add_base_args(parent_parser)
        
        return parser