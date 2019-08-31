# Dida Deep Learning Tools


## Package contents

This package is designed as a collection of fully compatible and flexible
patterns for deep learning projects. Rather than giving a detailed implementation
for every possible task 
("Why don't you have a *channel shuffle group transpose convolution layer*
for my Wasserstein DCGAN"), we rather aim to provide templates and best practices for
intuitive architectures.

Additionally, typical requirements such as Tensorboard logging, simple and extensible
training schemes, experiment tracking and model testing are covered. 

**All components of the package are to be understood as a template which is designed to be
easily modified - not a hard implementation of some specific model/layer/training**

See below for an
overview of the package contents:


```C++
.
├── config // generic project settings
│   ├── __init__.py
│   ├── config.py // gobal project config file
│   ├── device.py // device settings: cpu/cuda
├── docs // documentation files
├── layers // layer implementations
│   ├── __init__.py
│   └── conv.py // convolutional layers
├── models // model implementations
│   ├── __init__.py
│   └── unet.py // UNET
├── tests // unit tests
│   ├── test_models // testing of trained models
│   │   ├── __init__.py
│   │   └── test_unet.py // example test suite for UNET
│   └── __init__.py
├── training // implementation of generic training routines
│   ├── __init__.py
│   ├── fastai_training.py // fastai based template
│   ├── ignite_training.py // ignite based template
│   └── pytorch_training.py // raw pytorch based template
├── utils // utilities
│   ├── data // operations with data: custom loaders, wrappers, transformations
│   │   ├── __init__.py
│   │   └── datasets.py // pytorch dataset wrappers
│   ├── logging // custom logging routines
│   │   ├── __init__.py
│   │   └── csv.py // csv experiment tracker
│   ├── notify // training notification pipelines
│   │   ├── __init__.py
│   │   └── smtp.py // automated gmail notification via SMTP
│   ├── __init__.py
│   ├── loss.py // loss implementations
│   ├── name.py // file naming utilities
│   ├── path.py // path manipulation utilities
│   └── torchutils.py // generic pytorch tasks: forward passes, backward passes etc.
├── .gitignore
├── .pylintrc // linter settings
├── dev.yml // development and training requirements
├── Dockerfile // Docker image specs
├── environment.yml // inference and test time dependencies
├── hyperparameter.py // WIP 
├── Makefile // generic project tasks: cleaning, docs, building dependencies etc.
├── predict.py
├── README.md // this file
├── run_prediction.py
└── train.py
```

## Dependencies and environments

Dependencies are aggregated via `environment.yml` and `dev.yml` which specify a list of
`conda`-based project requirements. Run
```
make env
```
to only install dependencies needed for model testing and inference and run
```
make env-dev
```
to install all tool needed for training and corresponding development.
Note: `make env-dev` automatically runs `make env` as part of the recipe.


## Docker

A Docker image is provided by `Dockerfile`.
You need to have sufficient permissions on your machine and a working [Docker installation](https://docs.docker.com/install/overview/).
Run the image build process via 
```
docker build . -t <image-name>
```
and run a corresponding container via
```
docker run -it <image-name>
```
you can also directly use the make recipes `make docker-build` and `make docker-run`.

**NOTE**: Cuda compatibility is still to be fully integrated.
Pytorch itself is currently only providing a [deprecated GPU-compatible Docker image](https://github.com/pytorch/pytorch#docker-image)
via [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Until GPU compatibility is fully available for the Docker container, it is probably more interesting in the context of deploying and testing trained models
rather than training.

## Config files and project setup

The project uses a **global config file** which is found under `config.config.global_config`.
It specifies the project root as well as output destinations for weights, logs and experiment tracker files.
Whenever these parameters are changed, the training routines will automatically dump their
output to the corresponding directories.
When these directories do not exist, they are automatically created.

```python
global_config = {
    "ROOT_PATH": ROOT_PATH,
    "WEIGHT_DIR": WEIGHT_DIR,
    "LOG_DIR": LOG_DIR,
    "DATA_DIR": DATA_DIR
    }
```
Additionally, subcomponents such as training routines and model testing suites have their
own config files. These files specify a full set of parameters needed for the particular task.
As an example, as training run of the [ignite](https://pytorch.org/ignite/index.html)-based
training scheme in `training.ignite_training` contains a following **training config**:

```python
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
        "batch_size": 1,
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
```

Note that the config contains objects such as functions and classes which corresponding keyword arguments 
(for example `OPTIMIZER` and the corresponding `OPTIMIZER_CONFIG` with for example the learning rate).
The reason for this seemingly verbose architecture is that training routines can be fully captured and shared.
Additionally, they can easily be logged and no code refactoring is required to exchange hyperparameters.
Everything can happen in the config, once the training routine is set up as desired.

## Training


## Models


## Datasets


## Build documentation

