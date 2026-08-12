#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:27:40 2025

@author: bhirao
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from obspy import UTCDateTime as UTC
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pyocto
from obspy.geodetics import gps2dist_azimuth
import geopandas as gpd
from sklearn.linear_model import HuberRegressor
from scipy.optimize import least_squares

def get_cascadia_catalog():
    files = ['cascadia_m0_plus_2004_2008.csv',
             'cascadia_m0_plus_2008_2014.csv',
             'cascadia_m0_plus_2014_2018.csv',  
             'cascadia_m0_plus_2018_2025.csv']
    all_events = pd.DataFrame()
    for f in files:
        all_events = pd.concat([all_events, pd.read_csv('benchmark_data/' + f)])
    out = pd.DataFrame(all_events).sort_values(['time']).reset_index(drop = True)
    return out
    

def get_vmod(vm_id, vpvs = 1.75):
    vm = pd.read_csv('geographic/pnsn_vmods/' + vm_id + '.vel', sep = '\s+', header = None)
    vm = vm.rename({0 : 'vp', 1 : 'depth'}, axis = 1)
    vm['vs'] = vm['vp']/vpvs
    return vm

def random_sample_geographic(df, minlat, maxlat, minlon, maxlon, n_per_region = 10):
    # Define bin edges
    
    # Example DataFrame with lat, lon, depth
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Project to an equal-area CRS (example: EPSG:6933 = World Equidistant Cylindrical)
    gdf_proj = gdf.to_crs("EPSG:6933")  # You can choose based on region
    
    # Now create x, y columns in meters
    gdf_proj['x'] = gdf_proj.geometry.x
    gdf_proj['y'] = gdf_proj.geometry.y
    
    # Define bin widths in meters
    dx = 10_000  # 10 km
    dy = 10_000
    dz = 2_000  # 2 km depth bins
    
    # Bin in projected space
    gdf_proj['x_bin'] = (gdf_proj['x'] // dx).astype(int)
    gdf_proj['y_bin'] = (gdf_proj['y'] // dy).astype(int)
    gdf_proj['z_bin'] = (gdf_proj['depth'] // dz).astype(int)
    
    # Define 3D region ID
    gdf_proj['region'] = (
        gdf_proj['x_bin'].astype(str) + "_" +
        gdf_proj['y_bin'].astype(str) + "_" +
        gdf_proj['z_bin'].astype(str)
    )
    
    gdf_proj['region']
    
   # n_per_region = 5  # or however many you want
    
    sampled_df = (
    gdf_proj.groupby('region', group_keys=False)
            .apply(lambda x: x.sample(min(n_per_region, len(x)), random_state=42))
            )
    return sampled_df

def residuals(m, x, y):
    return y - m * x  # no intercept
    
#%%    

#MINDAYS = 365 * 8

# miniumum number of years for each station in each respective grid
# this many consecutive years within the 2002 to 2024 period
# the polygons are mapped:
# ['w1', 'w2', 'w3', 'e1', 'e2', 'e3']

ni_events = pd.read_csv('benchmark_data/ml_enhanced_catalog.csv')
all_event_locs = get_cascadia_catalog()

DX = 1

minyrs = [1,1,1,1,1,1]
vmods = ['p4', 'o4', 'nc', 'e3', 'o4', 'nc']
end_time = UTC(2024,1,1)
start_time = UTC(2002,1,1)

poly = json.load(open('geographic/cascadia_poly.json', 'r'))
ls_years = np.arange(2002,2024,1)

perm_stas = pd.read_csv('geographic/permanent_stations.txt', sep = '|')
perm_stas = perm_stas[perm_stas['Network ']!='AM'].reset_index(drop = True)
perm_stas['cont_days'] = [(UTC(x[' EndTime']) - UTC(x[' StartTime ']))/86400 for _, x in perm_stas.iterrows()]

# for ols to get vp/vs
#model = HuberRegressor(epsilon=1.35)  # epsilon is the tuning parameter

#perm_stas = perm_stas[perm_stas[' EndTime'] > UTC(2008,1,1)].reset_index(drop = True)
#%%
polygons = {}
vpvs_obs = []
for ii, ipoly in tqdm(enumerate(poly), total = len(poly)):
    
    #cvm_region = pd.read_csv('geographic/crescent_cvm_regions/' + ipoly['poly_id'] + '.csv')
    #np.unique(cvm_region['depth'])
    
    MINDAYS = minyrs[ii] * 365
    
    polygons[ipoly['poly_id']] = ipoly
    pll = (ipoly['minlon'], ipoly['minlat'])
    plr = (ipoly['maxlon'], ipoly['minlat'])
    pur = (ipoly['maxlon'], ipoly['maxlat'])
    pul = (ipoly['minlon'], ipoly['maxlat'])
    
    x0 = ipoly['olon']
    y0 = ipoly['olat']
    
    Iselect = (all_event_locs['latitude'] >= ipoly['minlat']) & \
            (all_event_locs['latitude'] <= ipoly['maxlat']) & \
            (all_event_locs['longitude'] >= ipoly['minlon']) & \
            (all_event_locs['longitude'] <= ipoly['maxlon'])
    region_events = all_event_locs[Iselect].reset_index(drop = True)
    sampled_events = random_sample_geographic(region_events, 
                                              ipoly['minlat'], 
                                              ipoly['maxlat'], 
                                              ipoly['minlon'], 
                                              ipoly['maxlon'],
                                              n_per_region = 10
                                              ).reset_index(drop = True)
    
    region_evid = sampled_events['id'].to_list()
    region_arrivals = pd.DataFrame([x for _,x in ni_events.iterrows() if x['event_id'] in region_evid])
    
    forwadati = region_arrivals[(~region_arrivals['P_arrival_time'].isnull()) & 
                             (~region_arrivals['S_arrival_time'].isnull())]
    
    wadati_data = {
                'origin' : [UTC(x) for x in forwadati['source_origin_time']],
                'Ptime' : [UTC(x) for x in forwadati['P_arrival_time']],
                'Stime' : [UTC(x) for x in forwadati['S_arrival_time']]
                }
    wadati_data['Ptt'] = np.array(wadati_data['Ptime']) - np.array(wadati_data['origin'])
    wadati_data['SPdt'] = np.array(wadati_data['Stime']) - np.array(wadati_data['Ptime'])
    
    
    # Define residual function with zero intercept
    xdat = np.float64(wadati_data['Ptt'])
    ydat = np.float64(wadati_data['SPdt'])

# Fit using robust Huber loss
    res = least_squares(residuals, x0=[1.0], loss='huber', f_scale=1.35, 
                        args=(xdat, ydat))

    slope = res.x[0]
    vpvs = 1+slope
    print("Slope (intercept = 0):", slope)
    
    vpvs_dat = {
                'region' : ipoly['poly_id'],
                'vpvs'   : vpvs
                }
    # try bootstrapping for uncertainties? may take a long time
    
    vpvs_obs.append(vpvs_dat)
    
    slope1 = 0.7
    slope2 = 0.85
    
    fig, ax = plt.subplots(figsize = (12,8))
    ax.scatter(xdat, ydat, s = 0.5)
    ax.plot(xdat, xdat*slope, color = 'black', ls = '--', label = 'solution Vp/Vs = ' + '%.3f' % (1+slope)) 
    ax.plot(xdat, xdat*slope1, color = 'black', ls = ':', label = 'Vp/Vs = ' + '%.3f' % (1+slope1)) 
    ax.plot(xdat, xdat*slope2, color = 'black', ls = '-.', label = 'Vp/Vs = ' + '%.3f' % (1+slope2)) 

    ax.legend()
    ax.set_title('Wadati ' + ipoly['poly_id'])
    
    ax.set_ylim(0,40)
    ax.set_xlim(0,50)
    
    ax.set_ylabel('S-P travel time')
    ax.set_xlabel('P travel time')
    
    fig.savefig('geographic/crescent_cvm_regions/' + ipoly['poly_id'] + '_' + 'wadati_vpvs.png', dpi = 300, bbox_inches = 'tight')
    
pd.DataFrame(vpvs_obs).to_csv('geographic/vpvs_ratios.csv')
    
    # next: make the 1d-averaged velocity models    
    