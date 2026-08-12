#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:24:09 2025

obtain and prep topo data

@author: bhirao
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy import interpolate
import pandas as pd
from tqdm import tqdm
import elevation
import math
import rasterio
from rasterio.merge import merge
from rasterio.transform import xy

def download_srtm_region_with_coords(lon_min, lat_min, lon_max, lat_max):
    """
    Downloads SRTM tiles from AWS for a bounding box, merges them,
    and returns the elevation array along with lat/lon coordinates.
    
    Parameters:
        lon_min, lat_min, lon_max, lat_max : float
            Bounding box in degrees (WGS84)
            
    Returns:
        dem_cropped : np.ndarray
            2D array of elevations at native resolution
        lat_grid : np.ndarray
            2D array of latitudes corresponding to each DEM pixel
        lon_grid : np.ndarray
            2D array of longitudes corresponding to each DEM pixel
    """
    # Compute tile coordinates
    lons = range(math.floor(lon_min), math.ceil(lon_max))
    lats = range(math.floor(lat_min), math.ceil(lat_max))

    # Generate AWS URLs
    urls = []
    for lat in lats:
        for lon in lons:
            ns = f"N{lat}" if lat >= 0 else f"S{-lat}"
            ew = f"W{-lon}" if lon <= 0 else f"E{lon}"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{ns}/{ns}{ew}.tif"
            urls.append(url)
    
    # Open tiles
  #  datasets = [rasterio.open(url) for url in urls]
    datasets = []
    for url in urls:
        try:
            dat = rasterio.open(url)
            datasets.append(dat)
        except: 
            continue
    
    # Merge tiles
    mosaic, out_trans = merge(datasets)
    dem = mosaic[0]  # single band
    
    # Crop to exact bounding box
    row_min, col_min = ~out_trans * (lon_min, lat_max)
    row_max, col_max = ~out_trans * (lon_max, lat_min)
    
    row_min, row_max = int(math.floor(row_min)), int(math.ceil(row_max))
    col_min, col_max = int(math.floor(col_min)), int(math.ceil(col_max))
    
    dem_cropped = dem[row_min:row_max, col_min:col_max]
    
    # Generate lat/lon grids for each pixel
    nrows, ncols = dem_cropped.shape
    rows = np.arange(row_min, row_max)
    cols = np.arange(col_min, col_max)
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon_grid, lat_grid = xy(out_trans, row_grid, col_grid)
    
    # Close datasets
    for ds in datasets:
        ds.close()
    
    return dem_cropped, lat_grid, lon_grid

# Function to fill NaN values using nearest neighbors
def fill_nan_with_nearest_neighbors(df, columns):
    # i think you can just loop this externally, feeding it only one depth slice at a time
   # for z_val in df['z'].unique():
   #     subset = df[df['z'] == z_val]
    subset = df
    for col in columns:
        mask = subset[col].notna()
        points = subset[mask][['x', 'y']].values
        values = subset[mask][col].values
        grid = subset[['x', 'y']].values
        filled_values = interpolate.griddata(points, values, grid, method='nearest')
        df.loc[:, col] = filled_values
    return df

regions = ['W1', 'W2', 'W3', 'E1', 'E2', 'E3']
grids = 'geographic/crescent_cvm_regions'

#outfile_root = 'geographic/crescent_cvm_simulps'

#%%

for ii, reg in enumerate(regions):
    #out_fileP = os.path.join(outfile_root, reg + '_simulps.P')
    dat = pkl.load(open(os.path.join(grids, reg + '_data.pkl'), 'rb'))
    nx, ny, nz = dat['coords']['shape']
    # unsure if i can use the x and y values from the origin 
    # instead of lat lon
    
    grid3d = dat['vp3d']
    minvel = np.nanmin(dat['vp']['Vp'])
    
   # Xgrid, Ygrid = np.meshgrid(xg, yg)
    
    longrid = dat['longrid']
    latgrid = dat['latgrid']
    
    bounds = (longrid.min()-(50/111.1), 
              latgrid.min()-(50/111.1),
              longrid.max()+(50/111.1),
              latgrid.max()+(50/111.1))
    
    #elevation.clip(bounds, output = os.path.join('geographic/crescent_cvm_regions/srtm/', reg + '.tif'))
    lon_min, lat_min, lon_max, lat_max = bounds
    dem, lat_grid, lon_grid = download_srtm_region_with_coords(lon_min, lat_min, lon_max, lat_max)

    
