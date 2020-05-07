"""Main pytorch lightning module for multi class segmentation"""

from lightning_modules.base_lightning_module import BaseModule
from utils.data.datasets import SegmentationDataset
from utils.preprocessing import get_preprocess, augmentation
from utils.loss import multi_class_smooth_dice_loss, multi_class_dice_ce_sum
from utils.cmap import GeoTIFFColourMap

import os
import torch
from torch import nn
import torchvision
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score

LOSSES = {
    "CE": nn.CrossEntropyLoss,
    "dice": objectise(multi_class_smooth_dice_loss),
    "dice_ce_sum": objectise(multi_class_dice_ce_sum)
}

class MultiClassSegmentationModule(BaseModule):
    """Lightning module for multi class semantic segmentation"""
    
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.loss = LOSSES[hparams.loss]()
            
        if self.adv_model is not None: # might be worth enabling?
            raise NotImplementedError("Adversarial loss cannot be used for segmentation.")
            
        if hasattr(hparams, "geotiff_colour_map"):
            self.cmap = GeoTIFFColourMap(hparams.geotiff_colour_map)
        else:
            self.cmap = None
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        self.unfreeze_layers()
        logdir = {"train_loss": loss}
        
        return {"loss": loss, "progress_bar": logdir, "log": logdir}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.model(x).float()
        
        loss = self.loss(y_hat, y)
        
        y_hat = y_hat.argmax(dim=1).cpu()
        y_hat = y_hat.view((y_hat.numel()))
        y = y.squeeze(1).int().cpu().view((y.numel()))
        
        acc = (y == y_hat).float().mean()
        f1 = torch.tensor(f1_score(y, y_hat, average="macro"))
        precision = torch.tensor(precision_score(y, y_hat, average="macro"))
        recall = torch.tensor(recall_score(y, y_hat, average="macro"))
        
        return {"val_loss": loss, "acc": acc, "f1": f1, "precision": precision, "recall": recall}
    
    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        accuracy = torch.stack([x['acc'] for x in outputs]).mean()
        f1 = torch.stack([x['f1'] for x in outputs]).mean()
        precision = torch.stack([x['precision'] for x in outputs]).mean()
        recall = torch.stack([x['recall'] for x in outputs]).mean()
        
        out_dict = {"val_loss": avg_val_loss, "val_acc": accuracy, "val_f1": f1, "val_precision": precision, "val_recall": recall}
        return {"log": out_dict, "val_loss": avg_val_loss}
    
    def on_epoch_end(self):
        """Write images to tensorboard for inspection"""
        
        if self.inspection_batch is None:
            x, y = next(iter(self._lazy_val_dataloader[0]))
            x = x[:min(32, len(x))]
            y = y[:min(32, len(x))]
        
            grid = torchvision.utils.make_grid(self._visualise(x))
            self.logger.experiment.add_image(f'initial_imgs', grid, 0)
            
            if self.cmap is None:
                batch = torch.sigmoid(y.float())
                for ch in range(batch.shape[1]):
                    grid = torchvision.utils.make_grid(batch[:, ch])
                    self.logger.experiment.add_image(f'ground_truth_channel {ch}', grid, self.current_epoch)
            
            elif isinstance(self.cmap, GeoTIFFColourMap):
                batch = self.cmap(y.detach().cpu().numpy())
                grid = torchvision.utils.make_grid(batch)
            
                self.logger.experiment.add_image(f'ground_truth', grid, self.current_epoch)
            else:
                raise NotImplementedError(f"Colour map type {type(self.cmap)} not implemented")            
            self.inspection_batch = x
        
        device = next(self.model.parameters()).device
        
        model_output = self.model(self.inspection_batch.to(device)).cpu()
        
        if self.cmap is None:
            batch = torch.sigmoid(model_output.float())
            for ch in range(batch.shape[1]):
                grid = torchvision.utils.make_grid(batch[:, ch])
                self.logger.experiment.add_image(f'predicted_output_channel {ch}', grid, self.current_epoch)
            
        elif isinstance(self.cmap, GeoTIFFColourMap):
            model_output = model_output.detach().argmax(dim=1).cpu().numpy()
            
            batch = self.cmap(model_output)
            grid = torchvision.utils.make_grid(batch)
            
            self.logger.experiment.add_image(f'predicted_output', grid, self.current_epoch)
        else:
            raise NotImplementedError(f"colour map type {type(self.cmap)} not implemented")
    
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