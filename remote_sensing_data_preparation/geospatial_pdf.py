"""Probabiltity density functions supported over the surface of the earth. Coordinates are output as (lat, lon) tuples (EPSG:4326)"""

import numpy as np
from shapely.geometry import Point

class UniformSpherePDF:
    """A PDF that uniformly samples over a sphere."""
    def __call__(self, n=1):
        lon = np.random.uniform(-180, 180, (n, 1))
        lat = np.arccos(2 * np.random.uniform(0, 1, (n, 1)) - 1.0) * (360 / (2 * np.pi)) - 90
        
        return np.concatenate([lat, lon], 1)

class UniformMultiPolygonPDF(UniformSpherePDF):
    """Uniformly sample a point contained within a series of polygons conatined in a geodataframe.
    
    Used for uniformly sampling on land or within a country (or set of countries) for example.
    """
    def __init__(self, geodataframe):
        self.polys = geodataframe.geometry.to_crs(4326)
    
    def __call__(self, n=1):
        pts = []
        for i in range(n):
            is_sea = True
            while is_sea:
                lat, lon = super().__call__(1)[0]
                is_sea = ~self.polys.contains(Point(lon, lat)).any()
                
            pts.append((lat, lon))
            
        return np.array(pts)
    
class DiscretePointsPDF:
    """Sample from a set of discrete points contained in geopandas dataframe"""
    def __init__(self, geodataframe):
        self.points = geodataframe.geometry.to_crs(4326)
    
    def __call__(self, n=1):
        return np.array([(pt.y, pt.x) for pt in self.points.sample(n)])