"""
Contains path, file and directory management helpers.
"""

import json
import os
import shutil

def create_dirs(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def remove_dirs(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

def init_directories(paths: list):
    for path in paths:
        create_dirs(path)

def read_json(path: str) -> dict:
    """
    Reads json from a given path and returns
    corresponding dictionary
    """
    with open(path) as file_content:
        config = json.load(file_content)
    return config
