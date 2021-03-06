"""
Contains basic project settings such as input/output directories etc.
"""

import os

#project root
ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

#output directories
WEIGHT_DIR = os.path.join(ROOT_PATH, "weights")
LOG_DIR = os.path.join(ROOT_PATH, "logs")

#data directories
DATA_DIR = os.path.join(ROOT_PATH, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
PREDICTED_DIR = os.path.join(DATA_DIR, "predicted")

global_config = {
    "ROOT_PATH": ROOT_PATH,
    "WEIGHT_DIR": WEIGHT_DIR,
    "LOG_DIR": LOG_DIR,
    "DATA_DIR": DATA_DIR
}
