"""Query, randomly select and download sentinel images."""

import configparser
from sentinelsat import SentinelAPI
from sentinelsat.sentinel import SentinelAPILTAError
import datetime
from shapely.geometry import Point
import random
from tqdm import tqdm

def get_sentinel(pts,
                 output_dir,
                 producttype,
                 start=datetime.date(2019, 1, 1),
                 end=datetime.date(2020, 1, 1),
                 **kwargs):
    
    """"Query sentinel images of type producttype intersecting pts and randomly select one between start and end"""
    
    config = configparser.ConfigParser()
    config.read("sentinel_scihub.ini")
    
    api = SentinelAPI(config["login"]["username"], config["login"]["password"])
    
    for pt in pts:
        products = api.query(area=f"{pt[0]}, {pt[1]}",
                             date=(start, end),
                             producttype=producttype,
                             **kwargs)
        
        if len(products) == 0:
            print("No products found, skipping")
            return False
        else:
            product_id = random.choice(list(products.keys()))
            print(product_id)
            
            try:
                api.download(product_id, directory_path=output_dir)
                return True
            except SentinelAPILTAError:
                print("product offline, skipping")
                return False