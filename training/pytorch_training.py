"""
A very simple raw pytorch training template including experiment tracking,
tensorboard logging and model checkpointing.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from tensorboardX import SummaryWriter

import torch
import torch.utils.data
from torch.nn import ELU

from config.config import global_config
from models.unet import UNET
from utils.logging.csv import write_log
from utils.loss import smooth_dice_loss, precision, recall, f1
from utils.path import create_dirs

train_config = {
    "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "SESSION_NAME": "training-run",
    "ROUTINE_NAME": sys.modules[__name__],
    "MODEL": UNET,
    "MODEL_CONFIG": {
        "ch_in": 12,
        "ch_out": 2,
        "n_recursions": 5,
        "dropout": .2,
        "use_shuffle": True,
        "activ": ELU
    },
    "DATA_LOADER_CONFIG": {
        "batch_size": 32,
        "shuffle": True,
        "pin_memory": True,
        "num_workers": 8
    },
    "OPTIMIZER": torch.optim.SGD,
    "OPTIMIZER_CONFIG": {
        "lr": 1e-3
    },
    "EPOCHS":  100,
    "LOSS": smooth_dice_loss,
    "METRICS": {
        "f1": f1,
        "precision": precision,
        "recall": recall
    },
    "DEVICE": torch.device("cuda"),
    "LOGFILE": "experiments.csv",
    "__COMMENT": None
}

def train(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
          training_config: dict=train_config, global_config: dict=global_config):
    """
    Template pytorch training routine. Takes a training and a test dataset wrapped
    as torch.utils.data.Dataset type and two corresponding generic
    configs for both gobal path settings and training settings. 
    """
    
    for path in global_config.values():
        create_dirs(path)

    # wrap datasets with Dataloader classes
    train_loader = torch.utils.data.DataLoader(train_dataset, **training_config["DATA_LOADER_CONFIG"])
    test_loader = torch.utils.data.DataLoader(test_dataset, **training_config["DATA_LOADER_CONFIG"])
           
    # model name & paths
    name = "_".join([train_config["DATE"], train_config["SESSION_NAME"]])
    modelpath = os.path.join(global_config["WEIGHT_DIR"], name)
    
    # instantiate model and optimizer
    model = training_config["MODEL"](**training_config["MODEL_CONFIG"]).to(training_config["DEVICE"])
    optimizer = training_config["OPTIMIZER"](model.parameters(), **training_config["OPTIMIZER_CONFIG"])
          
    # tensorboardX setup
    log_dir = os.path.join(global_config["LOG_DIR"], "tensorboardx", name)
    create_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    
    try:
        for epoch in range(training_config["EPOCHS"]):
            batch = 0
            for x, y in train_loader:
                
                x = x.to(training_config["DEVICE"], non_blocking=True)
                y = y.to(training_config["DEVICE"], non_blocking=True)
                
                optimizer.zero_grad()
                output = model(x)

                loss = training_config["LOSS"](output, y)
                loss.backward()
                optimizer.step()
                
                print("\repoch[{}] iteration[{}/{}] loss: {:.2f} "
                  "".format(epoch, batch, int(len(train_dataset) / training_config["DATA_LOADER_CONFIG"]["batch_size"]), 
                            loss, end=""))
                batch += 1
            
            # epoch checkpointing
            torch.save(model.state_dict(), modelpath + ".pth")
            print("Checkpoint saved to {}".format(modelpath + ".pth"))

    except KeyboardInterrupt:
        torch.save(model.state_dict(), modelpath +  ".pth")
        print("Model saved to {}".format(modelpath + ".pth"))
        raise KeyboardInterrupt
               
    # write weights
    torch.save(model.state_dict(), modelpath +  ".pth")
    print("Model saved to {}".format(modelpath + ".pth"))
    
    # write csv log file
    log_content = training_config.copy()
    #evaluator.run(test_loader)
    #log_content["VAL_METRICS"] = evaluator.state.metrics
    log_path = os.path.join(global_config["LOG_DIR"], training_config["LOGFILE"])
    write_log(log_path, log_content)
