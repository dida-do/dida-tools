"""Given a dataset of unlabelled geotiff files create targets from open source data for a variaty of tasks.

Functions should take an open rasterio dataset as an argument and return a numpy array ordered as CxHxW.

Not that this does not permit overlapping classes. These may be added in the future."""

import os
from pathlib import Path
import numpy as np
import cv2
import rasterio as rio
from rasterio import features
from rasterio import windows
from rasterio import warp
import geopandas as gpd
import osmnx as ox
from tqdm import tqdm
from argparse import ArgumentParser

import geospatial_pdf
from prepare_unlabelled_dataset import get_unlabeled_dataset
from paths import LAND_POLYGONS, LCLU_RASTER

def make_road_segmentation_target(src, all_touched=False, drop_tunnels=True, **kwargs):
    """Create a binary segmentation target for the detection of roads using OpenStreetMap.
    
    There are a numer of limitations here:
    * OpenStreetMap is incomplete and may have inaccuracies.
    * Clouds are not accounted for
    """
    
    target_img = np.zeros((src.height, src.width), dtype=np.uint8)
    
    try:
        feats = features.dataset_features(src, band=False, as_mask=True)
        bound = gpd.GeoDataFrame.from_features(feats, crs=4326)["geometry"][0]
        G = ox.graph_from_polygon(bound)
        nodes, edges = ox.graph_to_gdfs(G)
    
        if drop_tunnels:
            edges = edges.loc[edges["tunnel"].isna() | edges["tunnel"].astype(str).str.contains("no")]
    
        shapes = [(geom, 1) for geom in edges.to_crs(src.crs)["geometry"]]
    
        return features.rasterize(shapes=shapes, fill=0, out=target_img, transform=src.transform, all_touched=all_touched).astype(np.uint8)
    
    except:
        return target_img

def make_raster_segmentation_target(src, target_raster_path=LCLU_RASTER, **kwargs):
    """Make a segmentation target from a SINGLE BAND geotiff file.
    
    The default argument refers to the LCLU map available at https://land.copernicus.eu/local/riparian-zones/land-cover-land-use-lclu-image"""
    
    with rio.open(target_raster_path) as y_src:
        print(y_src.shape)
        print(src.shape)
        
        l, b, r, t = warp.transform_bounds(src.crs,
                                           y_src.crs,
                                           src.bounds.left,
                                           src.bounds.bottom,
                                           src.bounds.right,
                                           src.bounds.top)
        
        target_window = windows.from_bounds(l, b, r, t,
                                           transform=y_src.transform)
        
        # Not sure about the projection handling
        return cv2.resize(y_src.read(1, window=target_window), dsize=src.shape, interpolation=cv2.INTER_NEAREST)
    
def make_unpaired_target(pdf, save_dir, n_points=10, satellite_type="sentinel_2", **kwargs):
    """Create an unpaired image translation dataset s1 -> s2 or s2 -> s1"""
    
    if satellite_type == "sentinel_2":
        new_satellite = "sentinel_1"
    elif satellite_type == "sentinel_1":
        new_satellite = "sentinel_2"
    else:
        raise NotImplementedError(f"Satellite {satellite_type} not supported")
    
    get_unlabeled_dataset(pdf, save_dir, n_points, new_satellite)
    

TARGETS = {
    "road_segmentation": make_road_segmentation_target,
    "raster_segmentation": make_raster_segmentation_target
}

def make_dataset_targets(dataset_path: Path, target_type: str, satellite_type="sentinel_2", **kwargs):
    """Create targets for a dataset of geotiff files"""
    
    os.mkdir(dataset_path / "y")
    
    if target_type == "unpaired":
        # unpaired is a special case
        gdf = gpd.read_file(LAND_POLYGONS)
        
        pdf = geospatial_pdf.UniformMultiPolygonPDF(gdf)
        
        n_points = len([f for f in os.listdir(dataset_path / "x") if f.endswith(".tif")])
        
        make_unpaired_target(pdf, dataset_path / "y", n_points, satellite_type)
        return
    
    for f in tqdm(os.listdir(dataset_path / "x")):
        if f.endswith(".tif"):
            with rio.open(dataset_path / "x" / f) as src:
                target_img = TARGETS[target_type](src, **kwargs)
                
                with rio.open(dataset_path / "y" / f,
                              "w",
                              height=src.height,
                              width=src.width,
                              count=1,
                              dtype=np.uint8,
                              crs=src.crs,
                              transform=src.transform,
                              driver="GTiff") as dst:
                    
                    dst.write(target_img, 1)
                    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="dataset_path", required=True, type=Path)
    parser.add_argument("-t", dest="target_type", required=True, type=str)
    
    config = parser.parse_args()
    
    make_dataset_targets(config.dataset_path, config.target_type)