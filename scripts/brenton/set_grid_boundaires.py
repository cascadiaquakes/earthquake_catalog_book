#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:32:14 2025

set the grid boundaries for travel time here

also make a basic map of all of them

@author: bhirao
"""

import cartopy
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json
from pyproj import Proj, Transformer
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import subprocess


def write_json(new_data, filename='cascadia_poly.json'):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)  # Start with an empty list if the file doesn't exist
    
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data.append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)
        
        
def create_dict_std_param(poly_id, min_lon, max_lon, min_lat, max_lat):
    dictionary = {
        "poly_id": poly_id,
        "olon": (min_lon+max_lon)/2,
        "olat": (min_lat+max_lat)/2,
        "plat1": min_lat,
        "plat2": max_lat,
        "xsep": 2.0,
        "ysep": 2.0,
        "zmin": -4.0,
        "zmax": 100.0,
        "zsep": 1.0,
        "minlon": min_lon,
        "maxlon": max_lon,
        "minlat": min_lat,
        "maxlat": max_lat,
    }
    
    xmin, xmax, ymin, ymax = get_xy_bounds(min_lon, max_lon, min_lat, max_lat)
    # get these values from a separate function
    dictionary['xmin'] = xmin/1E3
    dictionary['xmax'] = xmax/1E3
    dictionary['ymin'] = ymin/1E3
    dictionary['ymax'] = ymax/1E3
    return dictionary

def get_xy_bounds(min_lon, max_lon, min_lat, max_lat):
    
    latlon1 = (max_lat, min_lon)
    latlon2 = (min_lat, max_lon)

    # Get all four corners of the rectangle
    lat_min = latmin1
    lat_max = latmax1
    lon_min = lonmin1
    lon_max = lonmax1

    corners = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_min),
        (lat_max, lon_max)
    ]

    # Define the Transverse Mercator projection
    # Example: UTM zone 33N (you might want to pick your own zone or define a custom TM)
    proj = Proj(proj='tmerc', lat_0=0, lon_0=(lon_min + lon_max)/2, ellps='WGS84')

    # Project all corners
    x_vals = []
    y_vals = []

    for lat, lon in corners:
        x, y = proj(lon, lat)
        x_vals.append(x)
        y_vals.append(y)

    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)

    print("x_min:", x_min, "x_max:", x_max)
    print("y_min:", y_min, "y_max:", y_max)
    
    return x_min, x_max, y_min, y_max


#%%

lonmin1 = -125.0
lonmin2 = -121.08
lonmax1 = -119.5
lonmax2 = -116.5

latmax1 = 49.25
latmax2 = 46.80
latmax3 = 44.0
latmin1 = 45.75
latmin2 = 43.0
latmin3 = 38.0

rectangles = [
    ['W1', lonmin1, latmin1, lonmax1, latmax1],
    ['W2', lonmin1, latmin2, lonmax1, latmax2],
    ['W3', lonmin1, latmin3, lonmax1, latmax3],
    ['E1', lonmin2, latmin1, lonmax2, latmax1],
    ['E2', lonmin2, latmin2, lonmax2, latmax2],
    ['E3', lonmin2, latmin3, lonmax2, latmax3]
 ]


# Creating DataFrame
df = pd.DataFrame(rectangles, columns=['poly_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat'])

if os.path.exists('cascadia_poly.json'):
    subprocess.run(['rm', 'cascadia_poly.json'])
    
for i, row in df.iterrows():
    write_json(create_dict_std_param(row.poly_id, row.min_lon, row.max_lon, row.min_lat, row.max_lat))


#%% # map of all 

#ax = plt.axes(projection=ccrs.Mercator())



lonmin0 = -125.5
lonmax0 = -116.0
latmin0 = 37.5
latmax0 = 50

lonticks = np.arange(lonmin0, lonmax0, (lonmax0 - lonmin0)/8)
latticks = np.arange(latmin0, latmax0, (latmax0 - latmin0)/8)

lonG, latG = np.meshgrid(lonticks, latticks)

projutm = Proj(proj = 'utm', zone = '10')
utm_E, utm_N = projutm(lonmin0, latmax0)
utm_W, utm_S = projutm(lonmax0, latmin0)

utm_xticks = np.arange(utm_E, utm_W, (utm_W-utm_E)/6)
utm_yticks = np.arange(utm_S, utm_N, (utm_N-utm_S)/8)

#conus_proj = ccrs.LambertConformal(central_longitude=-123,central_latitude=44)
#conus_proj = ccrs.UTM(10)
conus_proj = ccrs.Mercator()
#conus_proj = ccrs.epsg(4326)
pc_proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1,projection=conus_proj)
ax.coastlines()

ax.set_extent([lonmin0, lonmax0,  latmin0, latmax0])

#ax.add_feature(cfeature.BORDERS)
ax.set_xticks(lonticks, crs = ccrs.PlateCarree())
ax.set_yticks(latticks, crs = ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.OCEAN, facecolor='#CCFEFF')
ax.add_feature(cfeature.LAKES, facecolor='#CCFEFF')
ax.add_feature(cfeature.RIVERS, facecolor='#CCFEFF')
ax.add_feature(cfeature.LAND, facecolor='#FFE9B5')
state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='#FFE9B5')
ax.add_feature(state_borders, edgecolor='black')

# Convert them back to lon/lat
# These arrays will contain the geographic degrees corresponding to each tick
lon_labels = [pc_proj.transform_point(x, 0, conus_proj)[0] for x in ax.get_xticks()]
lat_labels = [pc_proj.transform_point(0, y, conus_proj)[1] for y in ax.get_yticks()]

# Format them as decimal degrees
ax.set_xticklabels([f"{lon:.2f}°" for lon in lon_labels], rotation = 30)
ax.set_yticklabels([f"{lat:.2f}°" for lat in lat_labels])

cols = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

markerstyles = ['--', ':']

for ii, ipoly in df.iterrows():
    
    ax.add_patch(mpatches.Rectangle(xy=[ipoly['min_lon'], ipoly['min_lat']], 
                                width=ipoly['max_lon'] - ipoly['min_lon'], 
                                height=ipoly['max_lat'] - ipoly['min_lat'],
                                facecolor='none',
                                alpha=1,
                                edgecolor = cols[ii],
                                transform=ccrs.PlateCarree(),
                                zorder = 1000,
                                ls = markerstyles[ii%2],
                                lw = 2)
                 )
                 
    xmid = ipoly['min_lon'] + (ipoly['max_lon'] - ipoly['min_lon'])/2
    ymid = ipoly['min_lat'] + (ipoly['max_lat'] - ipoly['min_lat'])/2
    ax.text(xmid, ymid, ipoly['poly_id'], transform = ccrs.PlateCarree(), 
            fontsize = 15,
            color = cols[ii],
            ha = 'center',
            va = 'center')
    print(xmid, ymid, ipoly['poly_id'])
    


plt.savefig('geographic/grids.png', dpi = 300, bbox_inches = 'tight')
# ax.set_yticks(latticks,
#               crs=conus_proj
#               )

# lat_formatter = LatitudeFormatter()
# ax.yaxis.set_major_formatter(lat_formatter)
#plt.plot([-120,-70],[35,45],linewidth=8, transform=ccrs.PlateCarree())

# for ii, ipoly in df.iterrows():
#     ax.add_patch(mpatches.Rectangle(xy=[-70, -45], width=90, height=90,
#                                 facecolor='blue',
#                                 alpha=0.2,
#                                 transform=ccrs.PlateCarree())
#              )

#plt.show()