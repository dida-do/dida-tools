"""
A very simple raw pytorch training template including experiment tracking,
tensorboard logging and model checkpointing.

A simple evaluation loop over a test dataset is implemented and
the best model with respect to this loss is saved while iterating
over the epochs. Additional test metric support can easily be incorporated.
"""

import os
import sys
from datetime import datetime

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
        "f1": f1,
        "precision": precision,
        "recall": recall
    },
    "DEVICE": torch.device("cuda"),
    "LOGFILE": "experiments.csv",
    "__COMMENT": None
}

def train(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
          training_config: dict = train_config, global_config: dict = global_config):
    """
    Template pytorch training routine. Takes a training and a test dataset wrapped
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
    name = "_".join([train_config["DATE"],
                     train_config["SESSION_NAME"]])
    modelpath = os.path.join(global_config["WEIGHT_DIR"], name)

    # instantiate model and optimizer
    model = training_config["MODEL"].to(training_config["DEVICE"])
    optimizer = training_config["OPTIMIZER"](model.parameters(),
                                             **training_config["OPTIMIZER_CONFIG"])

    # tensorboardX setup
    log_dir = os.path.join(global_config["LOG_DIR"], "tensorboardx", name)
    create_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    test_losses = []

    with log.Log(train_config=train_config, run_name=train_config['SESSION_NAME']) as logger:

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

                    logger.log_metric('Training Loss', loss.item())

                    print("\repoch[{}] iteration[{}/{}] loss: {:.2f} "
                          "".format(epoch,
                                    batch,
                                    int(len(train_dataset) / training_config["DATA_LOADER_CONFIG"]["batch_size"]),
                                    loss,
                                    end=""))
                    batch += 1

                # evaluation loop
                # NOTE: evaluation is performed w.r.t. model loss on chosen device,
                # all outputs are stored for global verification dataset loss
                with torch.no_grad():
                    y_vec = torch.tensor([]).to(training_config["DEVICE"], non_blocking=True)
                    y_hat_vec = torch.tensor([]).to(training_config["DEVICE"], non_blocking=True)
                    for x, y in test_loader:
                        x = x.to(training_config["DEVICE"], non_blocking=True)
                        y = y.to(training_config["DEVICE"], non_blocking=True)

                        model.eval()
                        output = model(x)
                        y_vec = y#torch.cat([y_vec, y])
                        y_hat_vec = output#torch.cat([y_hat_vec, output])

                # TODO tensorboard loss logging
                loss = training_config["LOSS"](y_hat_vec, y_vec)
                test_losses.append(loss)
                print(test_losses)

                #logging using the logging tool
                logger.log_metric('Evaluation Loss', loss.item())

                # best model checkpointing
                if torch.all(loss <= torch.stack(test_losses, dim=0)):
                    torch.save(model.state_dict(), modelpath + "bestmodel" + ".pth")
                    print("Best model saved to {}".format("bestmodel" + ".pth"))

                # epoch checkpointing
                torch.save(model.state_dict(), modelpath + ".pth")
                print("Checkpoint saved to {}".format(modelpath + ".pth"))

        except KeyboardInterrupt:
            torch.save(model.state_dict(), modelpath + ".pth")
            print("Model saved to {}".format(modelpath + ".pth"))
            raise KeyboardInterrupt

        # write weights
        torch.save(model.state_dict(), modelpath +  ".pth")
        print("Model saved to {}".format(modelpath + ".pth"))

        # write csv log file
        log_content = training_config.copy()
        log_content["VAL_LOSS"] = test_losses[-1]
        #evaluator.run(test_loader)
        #log_content["VAL_METRICS"] = evaluator.state.metrics
        log_path = os.path.join(global_config["LOG_DIR"], training_config["LOGFILE"])
        write_log(log_path, log_content)
        
    return log_content["VAL_LOSS"]