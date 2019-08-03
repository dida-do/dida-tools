"""
Contains path, file and directory management helpers.
"""

import os
from os import path
import shutil

def create_dirs(directory: str):
    if not path.exists(directory):
        os.makedirs(directory)

def remove_dirs(path: str):
    if path.exists(path):
        shutil.rmtree(path)

def init_directories(config: dict):
    for val in config.values():
        create_dirs(val)
