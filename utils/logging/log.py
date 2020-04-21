'''This module contains a wrapper for the mlflow logging framework. It will automatically set the 
local and global parameters for the training run when used in the deep learning tools, like architecture, learning rate etc. 
Mlflow will automatically create a folder called mlruns with the results. 
If you want a graphical overview use 'mlflow ui' (In a shell, not python). It will run a server on default port 5000,
which you can access in browser.

  Usage: 

from utils.logging.log import Log, Autolog

log = Log(training_config)
log.log_param('Architecture', 'CNN') 
log.log_metric('Loss', 0.1)

  or 

with Log(training_config) as log:
    log.log_metric('Loss', 0.1)
'''

import mlflow
from config.config import global_config
#from mlflow.pytorch import autolog

class Log():
    
    def __init__(self, 
                 training_config : dict = None,
                 run_id: str = None, 
                 experiment_id: str = None, 
                 run_name: str = None, 
                 nested: bool = False) -> None:
        
        mlflow.end_run()
        mlflow.start_run(run_id=run_id, 
                         experiment_id=experiment_id, 
                         run_name=run_name, 
                         nested=nested)
        mlflow.log_params(global_config)
        if training_config:
            mlflow.log_params(training_config)
        
    def log_param(self, key: str, value: str, *args) -> None:
        mlflow.log_param(key, value, *args)
        
    def log_params(self, params: {str}) -> None:
        mlflow.log_param(key, value, *args)
        
    def log_metric(self, key: str, value: str, *args) -> None:
        mlflow.log_metric(key, value, *args)
        
    def end_run(self) -> None:
        mlflow.end_run()
        
    def __enter__(self) -> None:
        return self
    
    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.end_run()
        
    def help(self) -> str:
        return """This Log module is a wrapper of mlflow. 
The Hyperparameters are values that differentiate the runs from each other and do not change during a run. 
Like the learning rate. 
Metrics are values that change due the run. Like the loss value.
        
There are versions of log_param() and log_metric() that log multiple values in a dictionary at once.
They are called log_params() and log_metrics() (s appended).
        
It makes sense to call the Log Object including the training_config provided in the training scripts.
Most important parameters are already included there and are useful to be logged. 

If you want to view the logged results you should start `mlflow ui` in a terminal.
Then you can access it via your browser. The default port is 5000

The runs will be saved in a directory called mlruns which will be created in the directory where you 
start the training run.

For more information check out the mlflow documentation."""
    
    
# Not in release version of MLFlow yet. Will be activated when it comes to stable. 
# Note that some frameworks like TensorFlow already work with autolog. See the mlflow docu to learn more. 
#class Autolog(Log):
#    '''Only implemented for pytorch here. Though there are mlflow integrations for other frameworks like TensorFlow.
#    Just import them instead of the pytorch one if you need them or directly use mlflow.'''
    
#    def __init__(self) -> str:
#        super().__init__()
#        autolog()
        
