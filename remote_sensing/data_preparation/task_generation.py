"""Download data and prepare a training task.

This takes a long time to run especially for sentinel 1 data as it performs an extremely complex processing pipeline.

For a program that involves sampling multiple tasks I would recomend running this in parallel to training the previous task. Nothing in here uses the GPU so it should be possible"""

import datetime
import geopandas as gpd
import shutil
from argparse import ArgumentParser
from pathlib import Path
import os

import geospatial_pdf
from prepare_unlabelled_dataset import get_unlabeled_dataset
from target_generation import make_dataset_targets, TARGETS
from geotiff2npy import convert_geotiff_folder

def create_dataset(task_name: str,
                   save_dir: Path,
                   satellite_type: str="sentinel_2",
                   pdf_name: str="uniform_land",
                   task: str="inpainting",
                   n_points: int=10,
                   start=datetime.date(2019, 1, 1),
                   end=datetime.date(2020, 1, 1),
                   max_cloud_cover: int=5,
                   block_size: int=256,
                   max_nodata: float=0.1,
                   delete_geotiffs: bool=True):
    
    """Download and prepare a dataset.
    
    Returns a path to the serialised numpy files
    
    :param task_name: Name to give the dataset
    :param save_dir: Root directory to save data under
    :param satellite_type: Satellite name. Currently `sentinel_1` and `sentinel_2` are supported
    :param pdf_name: Name of the probability distribution function to use. `uniform` and `uniform_land` supported
    :param task: pretraining task `inpainting`, `road_segmentation` and `raster_segmentation` supported.
    :param n_points: Number of files to download. Whole GeoTiffs, more npy files will be created.
    :param start: First possible date for image aquisition
    :param end: Last possible date for image aquisition
    :param max_cloud_cover: Maximum cloud cover for a usable image in percent. Ignored for sentinel_1
    :param block_size: size of each processed numpy file in pixels
    :param max_nodata: maximum proportion of a numpy file that can be missing
    :param delete_geotiffs: delete the geotiff files after numpy files are created.
    """
    
    geotiff_dir = save_dir / (task_name + "_gtiffs")
    numpy_dir = save_dir / (task_name + "_np")
    
    print("Initialising PDF")
    if pdf_name == "uniform":
        pdf = geospatial_pdf.UniformSpherePDF()
    elif pdf_name == "uniform_land":
        # This reads a shapefile containing the boundaries of land masses
        # available here https://osmdata.openstreetmap.de/data/land-polygons.html
        gdf = gpd.read_file("../data/simplified-land-polygons-complete-3857/simplified_land_polygons.shp")
        pdf = geospatial_pdf.UniformMultiPolygonPDF(gdf)
    else:
        raise NotImplementedError(f"PDF: {pdf_name} not implemented")
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    os.mkdir(geotiff_dir)
    os.mkdir(geotiff_dir / "x")
    
    # Get the GeoTiff dataset
    print("Getting Input Data")
    get_unlabeled_dataset(pdf,
                          geotiff_dir / "x",
                          n_points=n_points,
                          satellite_type=satellite_type,
                          max_cloud_cover=max_cloud_cover,
                          start=start,
                          end=end)
    
    # prepare targets
    if task == "inpainting":
        pass
    elif (task in TARGETS) or (task == "unpaired"):
        print("Preparing Targets")
        make_dataset_targets(geotiff_dir, task, satellite_type=satellite_type)
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
    
    # create numpy files
    print("converting to npy files")
    convert_geotiff_folder(geotiff_dir,
                           numpy_dir,
                           block_size=block_size,
                           max_nodata=max_nodata,
                           unpaired=(task == "unpaired"))
    
    if delete_geotiffs:
        shutil.rmtree(geotiff_dir)
    
    return numpy_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", dest="task_name", type=str, required=True)
    parser.add_argument("-o", dest="save_dir", type=Path, required=True)
    parser.add_argument("--satellite_type", dest="satellite_type", type=str, required=False, default="sentinel_2")
    parser.add_argument("--pdf_name", dest="pdf_name", type=str, required=False, default="uniform_land")
    parser.add_argument("--task", dest="task", type=str, required=False, default="inpainting")
    parser.add_argument("--n_points", dest="n_points", type=int, required=False, default=10)
    parser.add_argument("--keep_geotiffs", dest="keep_geotiffs", type=bool, required=False, default=False)
    
    config = parser.parse_args()
    
    create_dataset(config.task_name,
                   config.save_dir,
                   satellite_type=config.satellite_type,
                   pdf_name=config.pdf_name,
                   task=config.task,
                   n_points=config.n_points,
                   delete_geotiffs=~config.keep_geotiffs)