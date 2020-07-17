"""
Module to run unit test for learning successfully cifar100.
"""
import sys
from datetime import datetime
import unittest

# appending to path so python finds project modules
sys.path.append("../..")
sys.path.append("./")

from training import pytorch_training
from config.config import global_config
from utils.loss import precision, recall, f1

import torch
import torchvision
import torchvision.transforms as transforms

class TestLearning(unittest.TestCase):
    """
    """
    def setUp(self):
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        
        self.train_config = {
            "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "SESSION_NAME": "unittest-run",
            "ROUTINE_NAME": sys.modules[__name__],
            "MODEL": self.resnet,
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
            "DEVICE": torch.device("cuda"),
            "LOGFILE": "experiments.csv",
            "__COMMENT": None
        }
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.train_data = torchvision.datasets.CIFAR10(root='data', train=True,
                                                download=True, transform=transform)
        
        self.test_data = torchvision.datasets.CIFAR10(root='data', train=False,
                                               download=True, transform=transform)

    def test_learning(self):
        val_loss = pytorch_training.train(train_dataset=self.train_data,
                           test_dataset=self.test_data,
                           training_config=self.train_config,
                           global_config=global_config)

        assert val_loss < 2. # TODO is this a good bound?
