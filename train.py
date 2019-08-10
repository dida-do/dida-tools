"""
Contains a generic training interface
"""

import argparse

from config.config import read_json, global_config
from config.device import get_device

from models.unet import UNET
from training.pytorch import train
from utils.path import init_directories

def cli():
    #TODO
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
    