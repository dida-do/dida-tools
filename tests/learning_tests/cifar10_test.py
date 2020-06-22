import sys
from datetime import datetime

# appending to path so python finds project modules
sys.path.append("../..")

from training import pytorch_training
from config.config import global_config
from utils.loss import precision, recall, f1

import torch
import torchvision
import torchvision.transforms as transforms

resnet = torchvision.models.resnet18()
resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

train_config = {
    "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "SESSION_NAME": "training-run",
    "ROUTINE_NAME": sys.modules[__name__],
    "MODEL": resnet,
    "MODEL_CONFIG": {
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
    "EPOCHS": 1,
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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)

test_data = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=transform)

pytorch_training.train(train_dataset=train_data,
                       test_dataset=test_data,
                       training_config=train_config,
                       global_config=global_config)
