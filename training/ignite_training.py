"""
WIP - see
https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_tensorboardx.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from tensorboardX import SummaryWriter

import torch
import torch.utils.data
from torch.nn import ELU

from config.config import global_config
from models.unet import UNET
from utils.logging.csv import write_log
from utils.loss import smooth_dice_loss, precision, recall, f1
from utils.path import create_dirs

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

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
        "batch_size": 64,
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
        "f1": Loss(f1),
        "precision": Loss(precision),
        "recall": Loss(recall)
    },
    "DEVICE": torch.device("cuda"),
        "LOGFILE": "experiments.csv",
    "__COMMENT": None
}

# TODO add summary writer
# TODO add csv logging routine

def train(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
          training_config: dict=train_config, global_config: dict=global_config):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, **training_config["DATA_LOADER_CONFIG"])
    test_loader = torch.utils.data.DataLoader(test_dataset, **training_config["DATA_LOADER_CONFIG"])
    
    # instantiate model
    model = training_config["MODEL"](**training_config["MODEL_CONFIG"])
    
    optimizer = training_config["OPTIMIZER"](model.parameters(), **training_config["OPTIMIZER_CONFIG"])
    
    trainer = create_supervised_trainer(model=model, optimizer=optimizer, 
                                        loss_fn=training_config["LOSS"], device=training_config["DEVICE"])
    evaluator = create_supervised_evaluator(model, metrics=training_config["METRICS"], 
                                            device=training_config["DEVICE"])
      
        
    #TODO update variable names to training config key/value combinations
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % 2 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))

    trainer.run(train_loader, max_epochs=training_config["EPOCHS"])

