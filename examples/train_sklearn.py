"""
Contains a template command line interface for training.

As an example, the training.sklearn is used to train an SVM on MNIST.
This script should be run from the base directory.
"""

import argparse
import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from config.config import global_config
from utils.notify import smtp
from training import sklearn_training


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
        notifier = smtp.SMTPNotifier(args.smtp, args.smtp)

    if args.warnings:
        warnings.filterwarnings("ignore")

    #create datasets
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits["data"], digits["target"],
                                                        test_size=.1)

    sklearn_training.train(X_train,
                           X_test,
                           y_train,
                           y_test,
                           sklearn_training.train_config,
                           global_config)
