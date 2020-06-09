# Data Preparation

This directory contains code for downloading and preparing sentinel 1 and 2 datasets, additionally labels for dummy "pretraining" tasks can be created

Most files can also be run as self contained scripts as well.

Path constants are found in `paths.py` The files refered to in there should be downloaded and the paths changed to their location on your system.

* `prepare_unlabelled_dataset.py` downloads and processes a random dataset of GeoTiff files.
* `sentinel_1_preprocess.py` and `sentinel_2_preprocess.py` process SAFE files into GeoTiffs
* `target_generation.py` Creates labels for a GeoTiff dataset
* `task_generation.py` Puts it all together, randomly creating a dataset and creating targets. As well as producing `.npy` files for training.

## Additional Dependancies
* rasterio
* geopandas
* pyroSAR (And SNAP installed correctly!)
* osmnx

Sentinel api credentials should be placed in `sentinel_scihub.ini`.