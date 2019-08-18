"""
Contains a callback-based fastai.train.Learner template routine
including tensorboardX support, model checkpoints and csv logging
"""

import os
import sys
from pathlib import Path
from datetime import datetime

from fastai.callbacks.tracker import TerminateOnNaNCallback, SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.train import Learner
from fastai.basic_data import DataBunch

import torch
from torch.nn import ELU

from config.config import global_config
from models.unet import UNET
from utils.logging.csv import write_log
from utils.loss import smooth_dice_loss, precision, recall, f1
from utils.path import create_dirs

train_config = {
    "SESSION_NAME": "training-run",
    "ROUTINE_NAME": sys.modules[__name__],
    "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
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
        "batch_size": 64,
        "shuffle": True,
        "pin_memory": True,
        "num_workers": 8
    },
    "LR": 1e-3,
    "ONE_CYCLE": True,
    "EPOCHS":  100,
    "LOSS": smooth_dice_loss,
    "METRICS": [f1, precision, recall],
    "MIXED_PRECISION": True,
    "DEVICE": torch.device("cuda"),
    "LOGFILE": "experiments.csv",
    "__COMMENT": None
}

def train(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
          training_config: dict=train_config, global_config: dict=global_config) -> Learner:
    """
    Template training routine. Takes a training and a test dataset wrapped
    as torch.utils.data.Dataset type and two corresponding generic
    configs for both gobal path settings and training settings.
    Returns the fitted fastai.train.Learner object which can be
    used to assess the resulting metrics and error curves etc.
    """
    
    for path in global_config.values():
        create_dirs(path)

    # wrap datasets with Dataloader classes
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_config["DATA_LOADER_CONFIG"])
    test_loader = torch.utils.data.DataLoader(test_dataset, **train_config["DATA_LOADER_CONFIG"])
    databunch = DataBunch(train_loader, test_loader)

    # instantiate model and learner
    model = training_config["MODEL"](**training_config["MODEL_CONFIG"])
    learner = Learner(databunch, model, metrics=train_config["METRICS"],
                      path=global_config["ROOT_PATH"], model_dir=global_config["WEIGHT_DIR"],
                      loss_func=train_config["LOSS"])

    # model name & paths
    name = "_".join([train_config["DATE"], train_config["SESSION_NAME"]])
    modelpath = os.path.join(global_config["WEIGHT_DIR"], name)
     
    if train_config["MIXED_PRECISION"]:
        learner.to_fp16()

    learner.save(modelpath)

    torch.backends.cudnn.benchmark = True

    cbs = [
        SaveModelCallback(learner),
        LearnerTensorboardWriter(learner, Path(os.path.join(global_config["LOG_DIR"]), "tensorboardx") ,name),
        TerminateOnNaNCallback()
    ]

    # perform training iteration
    try:
        if train_config["ONE_CYCLE"]:
            learner.fit_one_cycle(train_config["EPOCHS"], max_lr=train_config["LR"], callbacks=cbs)
        else:
            learner.fit(train_config["EPOCHS"], lr=train_config["LR"], callbacks=cbs)
    # save model files
    except KeyboardInterrupt:
        learner.save(modelpath)
        raise KeyboardInterrupt
    
    learner.save(modelpath)
    val_loss = min(learner.recorder.val_losses)
    val_metrics = learner.recorder.metrics

    #write csv log file
    log_content = train_config.copy()
    log_content["VAL_LOSS"] = val_loss
    log_content["VAL_METRICS"] = val_metrics
    log_path = os.path.join(global_config["LOG_DIR"], train_config["LOGFILE"])
    write_log(log_path, log_content)

    return learner
    