"""Process ziped Sentinel 2 L1C SAFE files into geotiffs."""

import os
import shutil
import rasterio as rio
from zipfile import ZipFile
from tqdm import tqdm
import argparse
from pathlib import Path

NO_DATA_VAL = 0 # No data value for sentinel 2 L1C

def find_jp2_path(safe_root):
    """Find the path to the jp2 files in a standard S1 L1C SAFE format"""
    
    if not safe_root.endswith("/"):
        safe_root += "/"
    
    granule_dir = safe_root + "GRANULE/"
    return granule_dir + os.listdir(granule_dir)[0] + "/IMG_DATA/"

def processed_unzipped_safe(safe_root, out_dir):
    """Process a single SAFE directory into a geoTIFF"""
    bands = {}
    max_width = 0
    max_height = 0
    jp2_path = find_jp2_path(safe_root)
    
    for file in [f for f in os.listdir(jp2_path)]:
        if file[-7:-4] == "TCI":
            continue
            
        band = rio.open(jp2_path + file, driver="JP2OpenJPEG")
        max_width = max(max_width, band.width)
        max_height = max(max_height, band.height)
        bands[file[-6:-4]] = band
        
    idx = 1
    with rio.open(safe_root[:-4] + "tif",
                  "w",
                  driver="GTiff",
                  width=max_width,
                  height=max_width,
                  count=len(bands),
                  crs=bands["02"].crs,
                  transform=bands["02"].transform,
                  dtype=bands["02"].dtypes[0]) as dst:
        
        dst.nodata = NO_DATA_VAL
        
        for band_name in ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]:
            dst.write(bands[band_name].read(1), idx)
            idx += 1
            
            bands[band_name].close()
            
def process_all_safe_files(root_dir):
    """Unzip all zip files in a directory then process all safe files"""
    
    for f in tqdm(os.listdir(root_dir)):
        if f.endswith(".zip"):
            file_name = os.path.abspath(root_dir / f)
            print(f)
            print(file_name)
            
            with ZipFile(file_name) as zip_ref:
                zip_ref.extractall(root_dir)
                
            os.remove(file_name)
            
    for f in tqdm(os.listdir(root_dir)):
        if f.endswith(".SAFE"):
            file_name = os.path.abspath(root_dir / f)
            processed_unzipped_safe(file_name, root_dir)
            shutil.rmtree(file_name)
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="indir", required=True, type=Path)
    
    process_all_safe_files(parser.parse_args()["indir"])