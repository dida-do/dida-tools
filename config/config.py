"""
Contains basic project settings such as input/output directories etc.
"""

import os
import json

def read_json(path: str) -> dict:
    """
    Reads json from a given path and returns
    corresponding dictionary
    """
    with open(path) as file_content:
        config = json.load(file_content)
    return config

#project root
ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

#output directories
WEIGHT_DIR = os.path.join(ROOT_PATH, "weights")
CHECKPOINT_DIR = os.path.join(WEIGHT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_PATH, "logs")

#data directories
DATA_DIR = os.path.join(ROOT_PATH, "data")

config = {
    "ROOT_PATH": ROOT_PATH,
    "WEIGHT_DIR": WEIGHT_DIR,
    "CHECKPOINT_DIR": CHECKPOINT_DIR,
    "LOG_DIR": LOG_DIR,
    "DATA_DIR": DATA_DIR
}
