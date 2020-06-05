import os
import sys
from datetime import datetime
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

from config.config import global_config
from utils.path import create_dirs
from utils.logging.csvinterface import write_log
from utils.logging.log import Log

train_config = {
    "DATE": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "SESSION_NAME": "sklearn_training-run",
    "ROUTINE_NAME": sys.modules[__name__],
    "MODEL": SVC,
    "MODEL_CONFIG": {
        "C": 1.,
        "kernel": "rbf"
    },
    "LOSS": accuracy_score,
    "METRICS": {
        "accuracy": accuracy_score,
        "confusion_matrix": multilabel_confusion_matrix
    },
    "LOGFILE": "sklearn_experiments.csv",
    "__COMMENT": None
}

def train(X_train, X_test, y_train, y_test, train_config: dict=train_config,
          global_config: dict=global_config):

    # path from global config?
    for path in global_config:
        create_dirs(path)

    # model name and path
    name = "_".join([train_config["DATE"], train_config["SESSION_NAME"]])
    model_path = os.path.join(global_config["WEIGHT_DIR"], name)

    # instantiate model
    model = train_config["MODEL"](**train_config["MODEL_CONFIG"])

    # fit to training data
    model.fit(X_train, y_train)

    # dump model to disk
    joblib.dump(model, model_path+".joblib")

    # log metrics to csv
    predictions = model.predict(X_test)

    log_content = train_config.copy()
    log_content["VAL_LOSS"] = train_config["LOSS"](y_test, predictions)
    log_content["VAL_METRICS"] = {}

    for key, metric in train_config["METRICS"].items():
        log_content["VAL_METRICS"][key] = metric(y_test, predictions)

    log_path = os.path.join(global_config["LOG_DIR"], train_config["LOGFILE"])
    write_log(log_path, log_content)

    # log metrics to mlflow
    logger = Log(train_config=train_config, run_name=train_config["SESSION_NAME"])
    logger.log_metric("Test Loss", log_content["VAL_LOSS"])

    # return validation loss
    return log_content["VAL_LOSS"]
