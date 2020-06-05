"""Tools for hyperparameter optimisation using nevergrad"""

import nevergrad as ng
from dataclasses import dataclass
from config.config import global_config
import torch
from typing import Optional

from training.fastai_training import train as fastai_train
from training.ignite_training import train as ignite_train
from training.pytorch_training import train as pytorch_train
from training.lightning_training import train as lightning_train

def update_config(config, **kwargs):
    """Update a base configuration"""
    for key, value in kwargs.items():
        key_used = False
        if key in config:
            config[key] = value
            key_used = True
        else:
            for config_key, config_value in config.items():
                if isinstance(config_value, dict):
                    if key in config:
                        config[config_key][key] = value
                        key_used = True
            
        if not key_used:
            print(f"Argument {key} not found in config")
            
    return config

@dataclass
class TrainingContainer:
    """Callable class for training a network within nevergrad
    
    :param base_config: Base configuration for training
    :param train_dataset: Training dataset for non pytorch_lightning models
    :param test_dataset: Test dataset for non pytorch_lightning models
    :param global_config: Global training config
    :param backend: Which backend to use
    :param mode: Whether to mnimise of maximise the target metric"""
    
    
    base_config: dict
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    global_config: Optional[dict]=None
    backend: str="fastai"
    mode: str="max"
        
    def __post__init__(self):
        if self.global_config is None:
            self.global_config = global_config
            
        if self.mode == "max":
            self.coef = -1
        else:
            self.coef = 1
        
    def __call__(self, **kwargs):
        temp_config = update_config(self.config, **kwargs)
        
        if self.backend == "fastai":
            _, log_content, _ = fastai_train(self.train_dataset, self.test_dataset, temp_config, self.global_config)
            return self.coef * log_content["VAL_LOSS"]
        
        elif self.backend == "ignite":
            return self.coef * ignite_train(self.train_dataset, self.test_dataset, temp_config, self.global_config)
        
        elif self.backend == "pytorch":
            return self.coef * pytorch_train(self.train_dataset, self.test_dataset, temp_config, self.global_config)
        
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented. Use the specific class for lightning.")

@dataclass
class LightningTrainingContainer:
    """Callable class specifically for lightning"""
    config: dict
        
    def __call__(self, **kwargs):
        temp_config = update_config(self.config, **kwargs)
        return lightning_train(temp_config)
    
    
def hparam_opt(config: dict, hparam_space: dict, backend: str, train_dataset=None, test_dataset=None, global_config: dict=global_config, optimizer_name: str="OnePlusOne", budget: int=20, mode="max"):
    """Conduct a hypeparameter space search"""
    
    if backend == "lightning":
        training_function = LightningTrainingContainer(config)
    else:
        training_function = TrainingContainer(config, train_dataset, test_dataset, global_config, backend, mode=mode)
    
    hparam_vars = {key: ng.instrumentation.var.OrderedDiscrete(value) for key, value in hparam_space.items()}
    instrumentation = ng.Instrumentation(**hparam_vars)
    print(instrumentation)
    
    optimizer = ng.optimizers.registry[optimizer_name](instrumentation=instrumentation, budget=budget)
    
    return optimizer.minimize(training_function)