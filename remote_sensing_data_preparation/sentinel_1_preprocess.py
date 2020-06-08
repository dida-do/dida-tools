"""
Script and functions to batch preprocess level 1 sentinel-1 data into multi-band GeoTiff format.

Note that this does not split into batches or assemble traning pairs,
it only performs the SAR preprocessing to obtain a usable image

Usage:

`python sentinel1_preprocess.py -i <directory with sentinel-1 zips> -o <destination>`
"""

import os
from pathlib import Path
import shutil
from pyroSAR.snap import geocode
from pyroSAR import identify
from pyroSAR.auxdata import dem_autoload, dem_create
import rasterio
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List

def stack_geotiffs(file_list: List[Path], outfile: Path):
    """Stack a number of single band GeoTiffs into a single multi band one
    
    The files must all have the same metadata (projection etc)"""
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
        
    meta.update(count = len(file_list))
    
    with rasterio.open(outfile, "w", **meta) as dst:
        for i, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(i, src1.read(1))
                
                
def geocode_and_stack(infile: Path, outfile: Path, temp_dir: Path=Path("tmp/"), dem_type="AW3D30", **kwargs):
    """Process one sentinel 1 zip file. The main preprocessing is done inside geocode.
    
    In the future, this may be rewritten to allow more control over the processing graph"""
    
    
    bbox = identify(str(infile)).bbox()
    print("Downloading DEM files")
    vrt = dem_autoload(geometries=[bbox], demType=dem_type, vrt=f"{dem_type}.vrt", buffer=0.01)
    print("Converting DEM Files")
    dem_create(vrt, str(f"{dem_type}.tif"))
    
    geocode(str(infile), str(temp_dir), externalDEMFile=str(f"{dem_type}.tif"), **kwargs)
    os.remove(f"{dem_type}.tif")
    
    file_list = [str(file) for file in temp_dir.iterdir() if file.suffix == ".tif"]
    print("Stacking GeoTiffs")
    stack_geotiffs(file_list, str(outfile))
    shutil.rmtree(temp_dir)

def process_folder(indir):
    """Process all safe files in a folder"""
    for f in tqdm(indir.iterdir()):
        if f.suffix == ".zip":
            print(f)
            geocode_and_stack(f, f.with_suffix(".tif"))
            os.remove(f)
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="indir", required=True)
    
    process_folder(parser.parse_args()["indir"])