import sys
from datetime import datetime

import train
from training import pytorch_training
from config.config import global_config
from utils.loss import precision, recall, f1

import torch
import torchvision
import torchvision.transforms as transforms


train_config = {
    "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "SESSION_NAME": "training-run",
    "ROUTINE_NAME": sys.modules[__name__],
    "MODEL": torchvision.models.resnet18(),
    "MODEL_CONFIG": {
        "ch_in": 1,
        "ch_out": 10,
        "n_recursions": 5,
        "dropout": .2,
        "use_shuffle": True,
        "activ": torch.nn.ELU
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
    "EPOCHS":  1,
    "LOSS": torch.nn.CrossEntropyLoss(),
    "METRICS": {
        "f1": f1,
        "precision": precision,
        "recall": recall
    },
    "DEVICE": torch.device("cuda:1"),
    "LOGFILE": "experiments.csv",
    "__COMMENT": None
}


train_data = torchvision.datasets.MNIST(root='data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),
                                        target_transform=None, 
                                        download=False)

test_data = torchvision.datasets.MNIST(root='data', 
                                        train=False, 
                                        transform=transforms.ToTensor(),
                                        target_transform=None,
                                        download=False)


pytorch_training.train(train_dataset = train_data, 
                       test_dataset = test_data,
                       training_config = train_config,
                       global_config = global_config)




