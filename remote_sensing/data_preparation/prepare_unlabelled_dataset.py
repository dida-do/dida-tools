"""Randomly sample, download and process into GeoTiffs a Sentinel 1 or 2 dataset"""

from argparse import ArgumentParser
import datetime
import geopandas as gpd
import os
from pathlib import Path

from geospatial_pdf import UniformMultiPolygonPDF

from sentinel_api_handling import get_sentinel
from sentinel_1_preprocess import process_folder
from sentinel_2_preprocess import process_all_safe_files

from paths import LAND_POLYGONS

def get_files(coors, save_dir, satellite_type, start, end, max_cloud_cover):
    if satellite_type == "sentinel_1":
        
        exit_flag = get_sentinel(coors,
                                 save_dir,
                                 producttype="GRD", 
                                 sensoroperationalmode="SM IW EW", # Modes with both polarisation channels
                                 start=start,
                                 end=end)
        
        process_folder(save_dir)
        
    elif satellite_type == "sentinel_2":
        exit_flag = get_sentinel(coors, save_dir, producttype="S2MSI1C", start=start, end=end,
                                 cloudcoverpercentage=f"[0 TO {max_cloud_cover}]")
        process_all_safe_files(save_dir)
    else:
        raise NotImplementedError(f"Satellite {satellite_type} not found")

    return exit_flag

def get_unlabeled_dataset(pdf,
                          save_dir,
                          n_points=10,
                          satellite_type="sentinel_2",
                          max_cloud_cover=5,
                          start=datetime.date(2019, 1, 1),
                          end=datetime.date(2020, 1, 1)):
    
    """Download and process an unlabelled dataset from sentinel 1 or 2 given a probability distribution function over the surface of the earth"""
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    i = 0
    
    while i < n_points:
        coors = pdf(1)
        if get_files(coors, save_dir, satellite_type, start, end, max_cloud_cover):
            i += 1
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", dest="save_dir", default="../data/auto_dataset/", type=Path)
    parser.add_argument("--satellite_type", dest="satellite_type", default="sentinel_2")
    parser.add_argument("--max_cloud_cover", dest="max_cloud_cover", default=5)
    parser.add_argument("--n_points", dest="n_points", default=10, type=int)
    
    args = parser.parse_args()
    
    gdf = gpd.read_file(LAND_POLYGONS)
    pdf = UniformMultiPolygonPDF(gdf)
    
    get_unlabeled_dataset(pdf,
                          args.save_dir,
                          args.n_points,
                          args.satellite_type,
                          args.max_cloud_cover)