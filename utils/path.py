"""
Contains path, file and directory management helpers.
"""

import os
from os import path
import shutil

def make_dir_if_missing(directory: str):
    if not path.exists(directory):
        os.makedirs(directory)

def remove_if_exists(path: str):
    if path.exists(path):
        shutil.rmtree(path)
