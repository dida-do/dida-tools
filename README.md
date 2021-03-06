# dida tools

This package is intended as a collection of fully compatible and flexible
patterns for machine learning projects. Rather than giving a detailed implementation
for every possible task 
("Why don't you have a *channel shuffle group transpose convolution layer*
for my Wasserstein DCGAN"), we aim to provide templates and best practices for
intuitive architectures. 

Additionally, typical requirements such as Tensorboard logging, simple and extensible
training schemes, experiment tracking, hyperparameter optimization and model testing are covered.

If you use a very specific implementation of a layer/model/loss, you are invited to add your implementation here.
 
**All components of the package are to be understood as a template which is designed to be
easily modified - not a hard implementation of some specific pattern.**


## Installation, dependencies and environments

The installation of the package is very straight-forward. It simply needs to be cloned to a desired location with
```
git clone https://gitlab.com/didado/dida-tools.git
```
Subsequently, the environment needs to be set up. Dependencies are aggregated via `environment.yml` and `dev.yml` which specify a list of
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

Once the build was successful, the environment can be activate by running
```
conda activate dl-repo
```

## Package contents

The package contains several modules and sub-directories. **config** contains files with the major configurations. **layers** contains layer definitions for neural networks. So far, a variaty of convolutional layers is implemented. The layers can be used e.g. in **models**, where an implementation of e.g. U-Net can be found. **training** provides several training routines for a selection of frameworks, such as `fastai`, `ignite` or `lightning`. Additionally, tools for hyperparameter optimization are found here. **utils** collects various utilities, such as tools for conversions, logging, dataloaders and losses. Under **tests** unit tests are implemented. 

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
```

Note that the config contains objects such as functions and classes with corresponding keyword arguments 
(for example `OPTIMIZER` and the corresponding `OPTIMIZER_CONFIG` with the learning rate).
The reason for this seemingly verbose architecture is that training routines can be fully captured and shared.
Additionally, they can easily be logged and no code refactoring is required to exchange hyperparameters.
Everything can happen in the config, once the training routine is set up as desired.

## Training

Training is invoked by running `train.py` (or a similar script). The script needs to instantiate the training and validation dataset from the respective directories, e.g. `NpyDataset`s of `ndarray`s 
```python
from config.config import global_config
from training import pytorch_training

train_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'train/x'),
                                 os.path.join(global_config["DATA_DIR"], 'train/y'))

test_data = datasets.NpyDataset(os.path.join(global_config["DATA_DIR"], 'test/x'),
                                os.path.join(global_config["DATA_DIR"], 'test/y'))

```

Finally the specified training routine needs to be run, e.g. the standard `pytorch` training
```python
pytorch_training.train(train_data,
                       test_data,
                       pytorch_training.train_config,
                       global_config)

```

A selection of training routines is already implemented and stored in the module `training`. The signature of the training routines can vary. Of course, these steps can also be executed in a Jupyter Notebook.

## Models

Model implementations can be found in the `models` submodule. For now, an exemplary recursive U-Net implementation and binary classification models from `torchvision` are available. Feel free to add your own!

Additionally, `scikit-learn` provides a large number of algorithms.

## Experiment tracking

All experiments run by the different training routines allow for a unified way of tracking in 
a `.csv` file. This tracker files contains all experiment information such as date & time, model and model settings, 
user comments, location and name of the used training routine and the 
full information of the **training config** including all hyperparameters and used tools and their values (optimizers, losses, metrics...).

The tracker file is automatically created 
under `global_config["LOG_DIR"]/training_config["LOGFILE"]`. This path defaults to
`logs/experiments.csv`, if the config files are not changed. New training runs pointing to 
an already existing tracker file are simply appended to the table.

All training routines are compatible with the experiment tracker file, i.e. different training routines can dump
their information to a single tracker. Additionally, if a training routine is changed and new information is
added to the config, the same tracker can be used.
New columns are simply added whenever new hyperparameter names are added. If hyperparameters are removed from a training config file, the corresponding existing columns in the tracker file remain empty.

As an example, a test run with the `training.fastai-training` routine for the `models.unet.UNET` implementation automatically
creates the following entry in `logs/experiments.csv`:

```
,SESSION_NAME,ROUTINE_NAME,DATE,MODEL,MODEL_CONFIG,DATA_LOADER_CONFIG,LR,ONE_CYCLE,EPOCHS,LOSS,METRICS,MIXED_PRECISION,DEVICE,LOGFILE,__COMMENT,VAL_LOSS,VAL_METRICS,OPTIMIZER,OPTIMIZER_CONFIG
0,training-run,<module 'training.fastai_training' from '../training/fastai_training.py'>,20190818-183110,<class 'models.unet.UNET'>,"{'ch_in': 12, 'ch_out': 2, 'n_recursions': 5, 'dropout': 0.2, 'use_shuffle': True, 'activ': <class 'torch.nn.modules.activation.ELU'>}","{'batch_size': 1, 'shuffle': True, 'pin_memory': True, 'num_workers': 8}",0.001,1.0,1,<function smooth_dice_loss at 0x7f6380118d90>,"[<function f1 at 0x7f6380118f28>, <function precision at 0x7f6380118e18>, <function recall at 0x7f6380118ea0>]",1.0,cuda,experiments.csv,None,0.88317925,,,
```

## Tensorboard support

For all training routines, Tensorboard logging is supported via [TensorboardX](https://github.com/lanpa/tensorboardX),
which is a part of the `dev.yml` dependencies. To display the written logfiles, a working installation
of the original [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is needed (this is not contained in the dependencies).
Tensorboard is included in every full installation of TensorFlow.

By default, Tensorboard logs are dumped in `logs/tensorboardx/<name of the training run>`.
The logging root directory is contained in `config.config.global_config["LOG_DIR"]` and can be changed appropriately.
To access and visualize the logs, run 
```
tensorboard --logdir=logs/tensorboardx
```
in the default case - Tensorboard is now running on its default port `6006`, which you
can for example access via SSH port forwarding, if you are on a remote machine. Change the `logdir` argument accordingly whenever
you changed to logging directory settings. See the actual training routines for details about which information is logged.

## Datasets

In `utils.data.datasets` there are several implementations of dataloaders. The dataloaders hold the data directories and return image-label-pairs when `__getitem__` is called, i.e. the data is loaded from disk and augmented, if augmentations are required.
The plain-vanilla dataloader is `NpyDataset`, which assumes that all data points are stored in individual files. The data is loaded and converted to `ndarray`s. The files should be of one of the following formats

1. npy
1. pt / pth
1. csv
1. xls / xlsx / xlsm / xlsb / odf
1. hdf / h5 / hdf5 / he5
1. json
1. html

`utils.data.augmenter` implements an interface to apply image augmentations from `albumentations` seemlessly.

## Unit testing

Unit tests are found under `tests` and are written in standard `unittest` format. However, they can conveniently be run
using `pytest` and the recipe
```
make test
```
in the Makefile. Model unit tests under `tests.test_models` need a specific model and hyperparameters. The tests can be run on a given weight files or on a newly initialized model. See an example unit testing suite for the U-Net under `tests.test_models.test_unet` with the corresponding config.
Model unit tests contain training time testing (backprop steps etc.) as well as tests for predictions.

`tests.test_losses` implements basic sanity checks for loss functions and metrics. The functions are tested on randomly generated data and the tests cover the extreme cases (perfect match, complete mismatch) and the correct order (e.g. perfect match < half match < complete mismatch).

## Computer Vision
One major application of this set of packages is Computer Vision. To this end we provide proven model architectures and integrate albumentations, an extensive library for image augmentation.

In line with libraries such as opencv, skimage and albumentations we assume the images to be in channels-last ordering. Further we assume specific datatypes for different stages of the data processing pipeline.
The images can be stored on the hard drive in various formats and we support methods to load the data.
For pre-processing by the CPU we assume the images to be ndarrays with channels-last ordering. The dataloader should see to convert the images to torch.tensors with channels-first ordering once the images are send to the GPU.

## License
dida tools operates under the Apache License, version 2.0, as found in the LICENSE file. 
