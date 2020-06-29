"""Create a set of npy files from a set of GeoTiffs.

Reading GeoTiffs can be too slow to do on the fly in a training loop.

In production one would probably want to read directly from the GeoTiffs instead"""

from argparse import ArgumentParser
from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
from tqdm import trange, tqdm
from typing import Callable, Optional, List

@dataclass
class SingleGeoTiffIterable:
    """
    Note that this is not a pytorch dataset. It is an iterable to be used inside them.
    
    target_path, if provided should be a GeoTiff of the same size, resolution and projection containing the target"""
    file_path: Path
    block_size: int = 256
    max_nodata: float = 0.1
    target_path: Optional[Path] = None
    
    def __post_init__(self):
        with rio.open(self.file_path) as dataset:
            self.windows = []
            for i in range(dataset.width // self.block_size):
                for j in range(dataset.height // self.block_size):
                    window = Window(self.block_size * i, self.block_size * j, self.block_size, self.block_size)
                    mask = dataset.dataset_mask(window=window)
                    if (mask != 0).mean() > 1 - self.max_nodata:
                        self.windows.append(window)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        with rio.open(self.file_path) as dataset:
            x = dataset.read(window=self.windows[idx])
            
        if self.target_path is not None:
            with rio.open(self.target_path) as dataset:
                return x, dataset.read(window=self.windows[idx])
        else:
            return x
        
@dataclass
class MultiGeoTiffIterable:
    """
    Note that this is not a pytorch dataset. It is an iterable to be used inside them.
    
    target_dir_path, if provided should contain identically named GeoTiffs to dir_path"""
    dir_path: Path
    target_dir_path: Optional[Path] = None
    block_size: int = 256
    max_nodata: float = 0.1
    fname_list: Optional[List[str]] = None
        
    def __post_init__(self):
        self.subiters = []
        
        if self.fname_list is None:
            self.fname_list = os.listdir(self.dir_path)
        
        for f in tqdm(self.fname_list):
            if f.endswith(".tif"):
                if self.target_dir_path is not None:
                    target_path = self.target_dir_path / f
                else:
                    target_path = None
                
                it = SingleGeoTiffIterable(file_path=self.dir_path / f,
                                          target_path=target_path,
                                          block_size=self.block_size,
                                          max_nodata=self.max_nodata)
                
                self.subiters.append(it)
            
    def __len__(self):
        return sum([len(it) for it in self.subiters])
    
    def __getitem__(self, idx):
        """This currently iterates over all images in the folder until it reaches the required index. This can probably be improved to O(1) although this is not the slowest part of the process by far."""
        cumulative_size = 0
        for it in self.subiters:
            it_len = len(it)
            if idx - cumulative_size < it_len:
                return it[idx - cumulative_size]
            else:
                cumulative_size += it_len
                
        raise IndexError
        
def convert_geotiff_folder(dir_path: Path,
                           output_dir: Optional[Path]=None,
                           block_size=256,
                           max_nodata: float=0.1,
                           unpaired=True):
    
    """Convert a folder of GeoTiffs to a (much larger) folder of numpy arrays. Optionally also convert targets."""
    
    
    if output_dir is None:
        output_dir = dir_path.parent / (dir_path.name + "_np")
        
    output_dir.mkdir(parents=True)
    (output_dir / "x").mkdir()
    
    if not os.path.exists(dir_path / "y"):
        # Unlabelled setting
        it = MultiGeoTiffIterable(dir_path / "x", block_size=block_size, max_nodata=max_nodata)
        
        for i in trange(len(it)):
            np.save(output_dir / "x" / f"{i}.npy", it[i])
            
    elif unpaired == False:
        # Labelled setting
        (output_dir / "y").mkdir()
        it = MultiGeoTiffIterable(dir_path / "x", target_dir_path=dir_path / "y", block_size=block_size, max_nodata=max_nodata)
        
        for i in trange(len(it)):
            x, y = it[i]
            np.save(output_dir / "x" / f"{i}.npy", x)
            np.save(output_dir / "y" / f"{i}.npy", y)
    else:
        (output_dir / "y").mkdir()
        # Unpaired setting
        for folder in ["x", "y"]:
            print(folder)
            it = MultiGeoTiffIterable(dir_path / folder, block_size=block_size, max_nodata=max_nodata)
            for i in trange(len(it)):
                np.save(output_dir / folder / f"{i}.npy", it[i])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="dir_path", required=True, type=Path)
    parser.add_argument("-o", dest="output_dir", required=False, default=None, type=Path)
    parser.add_argument("--unpaired", dest="unpaired", required=False, default=False, type=bool)
    parser.add_argument("--block_size", dest="block_size", required=False, default=256, type=int)
    parser.add_argument("--max_nodata", dest="max_nodata", required=False, default=0.1, type=float)
    
    args = parser.parse_args()
    
    convert_geotiff_folder(args.dir_path,
                           args.output_dir, 
                           args.block_size,
                           args.max_nodata,
                           args.unpaired)