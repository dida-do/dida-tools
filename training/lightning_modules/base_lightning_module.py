"""Main pytorch lightning module"""

from argparse import ArgumentParser
from models.multi_channel_conv_nets import P2PModel, binary_clf_model
import os
from pathlib import Path
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from utils.torchutils import set_module_trainable

class BaseModule(pl.LightningModule):
    """Base module, to be inhereted by specific modules."""
    def __init__(self, hparams):
        super().__init__()
        
        torch.backends.cudnn.benchmark = True
        
        self.hparams = hparams
        
        if hasattr(hparams, "ch_out"):
            ch_out = hparams.ch_out
        else:
            ch_out = hparams.channels
        
        self.model = P2PModel(model_name=hparams.arch, ch_in=hparams.channels, ch_out=ch_out)
        
        set_module_trainable(self.model, False)
        
        if hparams.arch == "unet":
            set_module_trainable(self.model.model.out, True)
        else:
            set_module_trainable(self.model.model.classifier[-1], True)
            
        self.unfrozen = False
        
        if hasattr(self.hparams, "input_weights") and hparams.use_pretrained:
            try:
                self.load_model(self.save_dir.parents[0] / self.input_weights)
            except:
                print("pretrained weights not found")
        
        
        self.data_root = Path(hparams.data_root)
        
        all_fnames = os.listdir(self.data_root / "x")
        self.train_fnames, self.val_fnames = train_test_split(all_fnames, test_size=hparams.val_size)
        
        self.inspection_batch = None
        
        if hasattr(hparams, "adv_model"):
            adv_model = hparams.adv_model
                
            self.adv_model = binary_clf_model(adv_model, ch_out)
            set_module_trainable(self.adv_model, False)
            set_module_trainable(self.adv_model.fc, True)
            self.adv_loss_fn = nn.BCEWithLogitsLoss()
            
            if not hasattr(hparams, "w_adv"):
                print("No value for adversarial loss weight given. Using 0.5")
                self.w_adv = 0.5
            else:
                self.w_adv = hparams.w_adv
        else:
            self.adv_model = None
            self.w_adv = 0
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self):
        """Make sure to call in training_step"""
        if not self.unfrozen and self.current_epoch > self.hparams.epochs_til_unfreezing:
            self.unfrozen = True
            set_module_trainable(self.model, True)
            
            if self.adv_model is not None:
                set_module_trainable(self.adv_model, True)
                
    @staticmethod
    def get_optims(params, lr):
        opt = torch.optim.Adam(params, lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        
        return opt, sch
    
    def configure_optimizers(self):
        """If there is not advisarial model return one optimiser and one scheduler.
        
        If we have an adversarial model return an optimiser and scheduler for both the model and discriminator.
        Currently only adam and cosine annealing with the same learning rate for both models are implemented.
        """
        optimizer, scheduler = self.get_optims(self.model.parameters(), self.hparams.learning_rate)
        
        if self.adv_model is None:
            return [optimizer], [scheduler]
        else:
            optimizer_disc, scheduler_disc = self.get_optims(self.adv_model.parameters(), self.hparams.learning_rate)
            
            return [optimizer, optimizer_disc], [scheduler, scheduler_disc]
    
    def save_model(self, path):
        """Save a state dict only (Not including pytorch lightning specific data)"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load weights from a state dict. Ignore weights with sizes that do not match.
        
        This is used for loading weights from pretraining."""
        
        pretrained_dict = torch.load(path)
        model_dict = self.model.state_dict()
        
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if (k in model_dict) and (v.size() == model_dict[k].size())}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
    
    @staticmethod
    def __add_base_args(parent_parser):
        """Make sure to call in add_model_specific_args"""
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--data_root", type=str)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--arch", default="deeplabv3", type=str)
        parser.add_argument("--loss", default="L2", type=str)
        parser.add_argument("--channels", default=3, type=int)
        parser.add_argument("--val_size", default=0.1, type=float)
        parser.add_argument("--steps_til_unfreezing", default=10000, type=int)
        parser.add_argument("--no_data_aug", action="store_true")
        
        return parser
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """Method for adding arguments to an argparser for stand alone training"""
        raise NotImplementedError