"""
Contains a template command line interface for training.

As an example, the training.pytorch_training template routine is used
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
import pprint
import sys
import os
import warnings

from config.config import global_config
from config.device import get_device

from models.unet import UNET
from utils.data import datasets
from utils.notify import smtp

from training import pytorch_training


def cli():

    DESCRIPTION = """
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-s", "--smtp", help="Send SMTP mail notification", 
                        type=str)
    parser.add_argument("-w", "--warnings", action="store_true", 
                        help="Suppress all warnings")

    return parser.parse_args()

if __name__ == "__main__":
    
    args = cli()
    if args.smtp:
        notifier= smtp.SMTPNotifier(args.smtp, args.smtp)
        
    if args.warnings:
        warnings.filterwarnings("ignore")
    
    #create datasets
    train_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'train/x'),
                                     os.path.join(global_config["DATA_DIR"], 'train/y'))

    test_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'test/x'),
                                    os.path.join(global_config["DATA_DIR"], 'test/y'))

    #pass configuration and datasets to training routine
    learner, log_content, name = pytorch_training.train(train_data, test_data,
                                                       pytorch_training.train_config,
                                                       global_config)
    
    if args.smtp:
        pp = pprint.PrettyPrinter(indent=4)
        content = pp.pformat(log_content)
        notifier.notify(content, subject=name)
        