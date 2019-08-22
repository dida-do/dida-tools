"""
Contains a template command line interface for training.

As an example, the training.fastai_training template routine is used
in combination with the models.unet.UNET model.

The data loading is provided by a generic .npy array wrapper
in utils.datasets.NpyDataset, transformations and augmentation can be
added easily in the corresponding class.

We assume that the data is split into test and training examples
in the paths
<DATA_DIR>/train/x
<DATA_DIR>/train/y
<DATA_DIR>/test/x
<DATA_DIR>/test/y
where "x" corresponds to observation features and "y" corresponds to labels.
We additionally assume that in "x" and "y", the correspondence
is provided by files carrying the same file name. 
<DATA_DIR> is specified in config.config.global_config.

We do NOT provide a full Argparse suite, since every training routine
needs its own arguments provided by an external training config file/dictionary.
An example config is provided in training.fastai_training.
"""

import argparse
import sys
import os

from config.config import global_config
from config.device import get_device

from models.unet import UNET
from utils.data import datasets

from training import fastai_training

def cli(train_config):
    #TODO
    #parse generic config file names
    pass

if __name__ == "__main__":
        
    #create datasets
    train_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'train/x'),
                                     os.path.join(global_config["DATA_DIR"], 'train/y'))

    test_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'test/x'),
                                    os.path.join(global_config["DATA_DIR"], 'test/y'))

    #pass configuration and datasets to training routine
    learner = fastai_training.train(train_data, test_data,
                                    fastai_training.train_config, global_config)

    #notification routine
    #TODO
    