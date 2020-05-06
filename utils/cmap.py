import numpy as np
import rasterio
import torch

class GeoTIFFColourMap:
    """Extract the colourmap from a geotif file then use that to colour an array"""
    def __init__(self, geotif_file):
        with rasterio.open(geotif_file) as src:
            colour_dict = {k: (r, g, b) for k, (r, g, b, a) in src.colormap(1).items()}
            self.map = np.vectorize(colour_dict.get)
    
    def __call__(self, x):
        return torch.tensor(np.stack(self.map(x), axis=1)).squeeze(2)