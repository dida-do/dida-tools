# Dida Deep Learning Tools


## Package contents

This package is designed as a collection of fully compatible and flexible
patterns for deep learning projects. Rather than giving a detailed implementation
for every possible task 
("Why don't you have a *channel shuffle group transpose convolution layer*
for my Wasserstein DCGAN"), we rather aim to provide templates and best practices for
intuitive architectures.

Additionally, typical requirements such as tensorboard logging, simple and extensible
training schemes, experiment tracking and model testing are covered. 

**All components of the package are to be understood as a template which is designed to be
easily modified -- not a hard implementation of some specific model/layer/training**

See below for an
overview of the package contents:


```
.
├── config
│   ├── __init__.py
│   ├── config.py
│   ├── device.py
│   ├── hyperparameter.json
│   ├── predict.json
│   └── train.json
├── docs
├── layers
│   └── conv.py
├── models
│   ├── __init__.py
│   └── unet.py
├── tests
│   ├── test_models
│   │   ├── __init__.py
│   │   └── test_unet.py
│   └── __init__.py
├── training
│   ├── fastai_training.py
│   ├── ignite_training.py
│   └── pytorch_training.py
├── utils
│   ├── data
│   │   ├── __init__.py
│   │   └── datasets.py
│   ├── logging
│   │   ├── __init__.py
│   │   └── csv.py
│   ├── notify
│   │   ├── __init__.py
│   │   └── smtp.py
│   ├── __init__.py
│   ├── loss.py
│   ├── name.py
│   ├── path.py
│   └── torchutils.py
├── .gitignore
├── .pylintrc
├── dev.yml
├── Dockerfile
├── environment.yml
├── hyperparameter.py
├── Makefile
├── predict.py
├── README.md
├── run_prediction.py
└── train.py
```

## Dependencies and environments

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


## Config files and project setup


## Models

## Training

## Datasets


## Build documentation

