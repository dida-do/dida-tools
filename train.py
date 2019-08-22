"""
Contains a template command line interface for training.

As an example, the training.fastai_training template routine is used
in combination with the models.unet.UNET model.

The data loading is provided by a generic .npy array wrapper
in utils.datasets.NpyDataset, transformations and augmentation can be
added easily in the corresponding class.

We assume that the data is split into test and training examples
in the paths
<data_root>/train/x
<data_root>/train/y
<data_root>/test/x
<data_root>/test/y
where "x" corresponds to observation features and "y" corresponds to labels.
We additionally assume that in "x" and "y", the correspondence
is provided by files carrying the same file name.

We do NOT provide a full Argparse suite, since every training routine
needs its own arguments provided by an external training config file/dictionary.
An example config is provided in training.fastai_training.
"""

import argparse

from config.config import read_json, global_config
from config.device import get_device

from models.unet import UNET
from training.pytorch import train
from utils.path import init_directories

def cli(train_config):
    pass

if __name__ == "__main__":

    #parse command line args into configs
    args = cli()

    #get configs
    model_config = read_json(args.model_config)
    train_config = read_json(args.train_config)
    device = get_device()

    #update config settings with cli arguments
    #TODO

    #initialize directories
    init_directories(global_config.values())

    #wrap dataset and provide iterable -> dataloader class
    dataloader = Dataloader(dataset)

    #instantiate model
    model = UNET(**model_config)

    # throw iterable and training config and global config
    # into training routine
    train(model, dataloader, train_config, global_config, device)

    #notification routine
    if args.notify:
        #TODO
        pass
    