"""
Command line interface for simple batch prediction.
"""

import argparse
import numpy as np
import torch
import tqdm
import os

from models.unet import UNET
from utils.data.datasets import NpyPredictionDataset
from utils.path import create_dirs
from utils.torchutils import load_model, forward

predict_config = {
    "PREFIX": "",
    "SUFFIX": "",
    "MODEL": UNET,
    "MODEL_CONFIG": {
        "ch_in": 12,
        "ch_out": 2,
        "n_recursions": 5,
        "use_shuffle": True,
        "activ": torch.nn.ELU },
    "WEIGHTS": "/path/to/model/weights.pth",
    "DEVICE": torch.device("cuda"),
    }


def cli():

    DESCRIPTION = """
    Command line interface for batch compatible generic model prediction.

    Usage:
        $ python predict.py -i path/to/my/files/*.npy -o my/output/path
        
    Performs predictions for all .npy files obtained through shell globbing
    and serialises the outputs as specified in the main routine below.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-i", "--input", type=str, help="Input file paths",
                        required=True, nargs="+")
    parser.add_argument("-o", "--output", type=str, help="Output directory",
                        required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    create_dirs(args.output)
    
    # create prediction dataset
    dataset = NpyPredictionDataset(args.input) 
    
    # instantiate model
    model = load_model(predict_config["MODEL"], predict_config["MODEL_CONFIG"], 
                       predict_config["WEIGHTS"], predict_config["DEVICE"])
    
    # perform predictions
    for path, data in tqdm.tqdm(dataset):
        
        # note: shape manipulation is needed since we do not feed batches, but single images
        prediction = forward(model, data[None, :, :, :], predict_config["DEVICE"])
        
        # get output file basename
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        
        # insert your file serialisation routine based on new name and output path here
        # example write file as .npy - note that if CUDA is used, the tensor needs to be transferred to CPU:
        out_name = "{}{}{}.npy".format(predict_config["PREFIX"], name, predict_config["SUFFIX"])
        out_path = os.path.join(args.output, name)
        np.save(out_path, prediction.cpu().numpy())   
    