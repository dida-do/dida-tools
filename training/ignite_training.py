"""
An ignite-based template training routine.
Contains automated checkpointing, csv routine history logging and
general tracking of metrics/tensorboard support via tensorboardX.
"""

import os
import sys
from datetime import datetime

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

from tensorboardX import SummaryWriter

import torch
import torch.utils.data
from torch.nn import ELU

from config.config import global_config
from models.unet import UNET
from utils.logging.csvinterface import write_log
from utils.loss import smooth_dice_loss, precision, recall, f1
from utils.path import create_dirs
import utils.logging.log as log

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
        "f1": Loss(f1),
        "precision": Loss(precision),
        "recall": Loss(recall)
    },
    "DEVICE": torch.device("cuda"),
    "LOGFILE": "experiments.csv",
    "__COMMENT": None
}

def train(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
          training_config: dict = train_config, global_config: dict = global_config):
    """
    Template ignite training routine. Takes a training and a test dataset wrapped
    as torch.utils.data.Dataset type and two corresponding generic
    configs for both gobal path settings and training settings.
    """

    for path in global_config.values():
        create_dirs(path)

    # wrap datasets with Dataloader classes
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               **training_config["DATA_LOADER_CONFIG"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              **training_config["DATA_LOADER_CONFIG"])

    # model name & paths
    name = "_".join([train_config["DATE"], train_config["SESSION_NAME"]])
    modelpath = os.path.join(global_config["WEIGHT_DIR"], name)

    # instantiate model
    model = training_config["MODEL"](**training_config["MODEL_CONFIG"])

    optimizer = training_config["OPTIMIZER"](model.parameters(),
                                             **training_config["OPTIMIZER_CONFIG"])

    # set up ignite engine
    training_config["METRICS"].update({"loss" : Loss(training_config["LOSS"])})
    trainer = create_supervised_trainer(model=model, optimizer=optimizer,
                                        loss_fn=training_config["LOSS"],
                                        device=training_config["DEVICE"])
    evaluator = create_supervised_evaluator(model,
                                            metrics=training_config["METRICS"],
                                            device=training_config["DEVICE"])


    # tensorboardX setup
    log_dir = os.path.join(global_config["LOG_DIR"], "tensorboardx", name)
    create_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    # log using the logging tool
    logger = log.Log(training_config, run_name=train_config['SESSION_NAME'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        if iteration % 4 == 0:
            print("\repoch[{}] iteration[{}/{}] loss: {:.2f} ".format(engine.state.epoch,
                                                                      iteration, len(train_loader),
                                                                      engine.state.output), end="")

    # generic evaluation function
    def evaluate(engine, loader):
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        return metrics

    # training data metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        print("\ntraining results - epoch {}".format(engine.state.epoch))
        metrics = evaluate(engine, train_loader)
        print(metrics)
        for key, value in metrics.items():
            logger.log_metric(key, value)
            writer.add_scalar("training/avg_{}".format(key), value, engine.state.epoch)

    # test data metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        print("test results - epoch {}".format(engine.state.epoch))
        metrics = evaluate(engine, test_loader)
        print(metrics)
        for key, value in metrics.items():
            writer.add_scalar("validation/avg_{}".format(key), value, engine.state.epoch)

    # model checkpointing
    @trainer.on(Events.EPOCH_COMPLETED)
    def model_checkpoint(engine):
        torch.save(model.state_dict(), modelpath + ".pth")
        print("Checkpoint saved to {}".format(modelpath + ".pth"))

    # training iteration
    try:
        trainer.run(train_loader, max_epochs=training_config["EPOCHS"])
    except KeyboardInterrupt:
        torch.save(model.state_dict(), modelpath +  ".pth")
        print("Model saved to {}".format(modelpath + ".pth"))
        raise KeyboardInterrupt

    # write weights
    torch.save(model.state_dict(), modelpath +  ".pth")

    # write csv log file
    log_content = training_config.copy()
    evaluator.run(test_loader)
    log_content["VAL_METRICS"] = evaluator.state.metrics
    log_path = os.path.join(global_config["LOG_DIR"], training_config["LOGFILE"])
    write_log(log_path, log_content)

    logger.end_run()
    
    return evaluator.state.metrics["training/avg_loss"]
