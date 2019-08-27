# Dida Deep Learning Tools


## Package contents

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


## Build documentation



```
├──  .dvc                 - clean DVC config
├──  .gitlab-ci.yml       - Gitlab CI template file
├──  .pylintrc            - contains linter settings
├──  Dockerfile           - project template image for Docker
├──  environment.yml      - full requirement specs for all contents
├──  Makefile             - contains generic project tasks (tests, docs, cleaning...)
├──  README.md            - this file
│ 
├──  train.py             - command line interface for generic training routines 
├──  predict.py           - command line interface for generic inference routines
├──  hyperparameter.py    - command line interface for generic hyperparameter search
│ 
├──  config
│    ├── config.py        - default config file.
│    └── device.py        - CPU/GPU settings.
│
├──  data                 - contains datasets, loaders and preprocessing tools  
│    ├── datasets         - contains the actual train/test data
│    ├── dataloader       - contains data wrapper classes and loading routines
│    └── transforms       - routines and tools for data preprocessing and augmentation
│
├──  logging              - contains training routines
│    ├── logger.py        - logging template
│    └── tboardx.py       - wrappers for tensorboardx
|
├──  hyperparameter       - contains hyperparameter search routines
│    └── nevergrad.py     - contains a nevergrad-based template optimization procedure
|
├──  training             - contains training routines
│    └── fastai.py        - contains a fastai-based template training procedure
│
├── layers                - custom layer architectures
│    └── conv_layer.py
│
├── models                - contains model class implementations
│    └── unet.py
|
├── notification          - contains notification and scheduling routines
│    └── smtp.py          - SMTP mail notifier
│
├── solver                - this folder contains optimizer of your project.
│    └── lr_scheduler.py 
│ 
├── utils
|    ├── name.py          - string manipulation and model name generation routines
|    ├── path.py          - path manipulation helpers
|    ├── checkpoint.py    - model saving routines
│    └── build.py         - model compiling and JIT helpers
│ 
├──test                   - contains unit tests
|    ├── utils.py         - unit test related utilities
|    └── tests            - unit test cases 
|         └── [...]
|
├──  weights              - default output directory
|    └── tests            - default checkpoint output directory
```