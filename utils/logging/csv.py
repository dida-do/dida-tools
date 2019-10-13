"""
Contains routines for .csv file expriment logging.
"""

import os
import pandas as pd

def config2series(config: dict) -> pd.Series:
    """
    Takes a config dictionary, flattens it on
    the first level by converting all objects to strings
    and returns it as a pandas series with keys as
    corresponding indices.
    """
    flat = {key: str(val) for key, val in config.items()}
    return pd.Series(flat)

def write_log(filepath: str, config: dict, verbose=True) -> None:
    """
    Appends information from config dictionary
    to .csv log file specified by path.
    If the file does not exist, a new file is created.
    The config dictionary is flatted on the first level
    by converting all contained information to strings.
    Whenever a new key occurs is it appended at the last column of
    the existing logfile.
    """
    series = config2series(config)
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, index_col=0)
        df = df.append(series, ignore_index=True)
        cols = df.columns
    else:
        if verbose:
            print("logfile generated: {}".format(filepath))
        df = pd.DataFrame(series).T
        cols = df.columns
    df[cols].to_csv(filepath)
