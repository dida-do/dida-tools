"""Minimal lightning training function"""

import pytorch_lightning as pl
from argparse import Namespace

from training.lightning_modules.binary_segmentation_module import BinarySegmentationModule
from training.lightning_modules.multi_class_segmentation_module import MultiClassSegmentationModule

train_config = {
    "arch": "unet",
    "gpus": 1,
    "use_16_bit": True,
    "val_size": 0.1,
    "channels": 13,
    "num_workers": 8,
    "max_val": 6000,
    "exp_name": "test",
    "task": "multiclass_segmentation",
    "batch_size": 16,
    "learning_rate": 1e-4,
    "min_nb_epochs": 10,
    "max_nb_epochs": 100,
    "loss": "dice_ce_sum",
    "data_root": "path/to/data",
    "monitor": "val_f1",
    "mode": "max",
    "patience": 10,
    "ch_out": 256,
    "output_weights": "weights.pt",
    "no_data_aug": False,
    "epochs_til_unfreezing": 0,
}

modules = {
    "binary_segmentation": BinarySegmentationModule,
    "multiclass_segmentation": MultiClassSegmentationModule
}

def train(training_config: dict=train_config):
    """Wrapper function for pytorch lightning training"""
    
    hparams = Namespace(**training_config)
    module = modules[hparams.task](hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         use_amp=hparams.use_16_bit,
                         min_epochs=hparams.min_nb_epochs,
                         max_epochs=hparams.max_nb_epochs)
    
    trainer.fit(module)
    
    return trainer.checkpoint_callback.best