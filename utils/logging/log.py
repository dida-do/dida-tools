'''This module contains a wrapper for the mlflow logging framework. It will automatically set the 
local and global parameters for the training run when used in the deep learning tools. 
Mlflow will automatically create a folder called mlruns with the results. 
If you want a graphical overview use 'mlflow ui' (In a shell, not python). It will run a server on default port 5000. 
'''

import mlflow
import os
import config.config as global_config

#TODO:
#get training_config

class Log():
    
    def __init__(self, 
                 run_id: str = None, 
                 experiment_id: str = None, 
                 run_name: str = None, 
                 nested: bool = False):
        
        mlflow.start_run(run_id=run_id, 
                         experiment_id=experiment_id, 
                         run_name=run_name, 
                         nested=nested)
        mlflow.log_params(global_config)
        mlflow.log_params(training_config)
        
    def log_param(self, key: str, value: str, *args):
        mlflow.log_param(key, value, *args)
        
    def log_params(self, params: {str}):
        mlflow.log_param(key, value, *args)
        
    def log_metric(self, key: str, value: str, *args):
        mlflow.log_metric(key, value, *args)
        
    def end_run(self)
        mlflow.end_run()
        
    def __enter__(self):
        self.__init__()
    
    def __exit__(self):
        self.end_run()
        
    def help(self):
        pass
    
    
class Autolog(Log):
    
    def __init__(self):
        pass
        