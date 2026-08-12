#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:15:14 2026

combined postprocessing of nlloc catalog

correct_topo.py -> merge_catalogs.py -> 

@author: bhirao
"""

import pandas as pd 
import pickle as pkl
from scipy.interpolate import griddata
import os
import math
import matplotlib.pyplot as plt
from obspy import UTCDateTime as UTC
import json
import shapely
import geopandas as gpd
import os
import pickle as pkl
import pandas as pd
import json
import numpy as np
from functools import reduce
import math
from obspy.geodetics import gps2dist_azimuth
from sklearn.neighbors import BallTree
import networkx as nx
from obspy import UTCDateTime as UTC
import glob
import sys
sys.path.append('/gpfs/projects/amt/shared/cascadia_data_mining/MSH/nonlinloc_code')
import nonlinloc_tools
from collections import Counter
import tqdm
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely import make_valid


def haversine_distance(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """
    Calculate the Haversine distance between two points on the Earth (specified in decimal degrees).
    
    Parameters:
        lat1, lon1: Latitude and Longitude of point 1 (in decimal degrees)
        lat2, lon2: Latitude and Longitude of point 2 (in decimal degrees)
        radius_km: Radius of the Earth (default 6371.0 km)
    
    Returns:
        Distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    
    # Haversine formula
    dφ = φ2 - φ1
    dλ = λ2 - λ1

    a = math.sin(dφ / 2.0)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return radius_km * c

def remove_edge_events(df_reg, geodat, min_dist):
    # min_dist is the distance from an edge--below which is removed from the catalog
    minx, maxx = geodat['coords']['xvals'].min(), geodat['coords']['xvals'].max()
    miny, maxy = geodat['coords']['yvals'].min(), geodat['coords']['yvals'].max()
    I = (df_reg['x'] > (minx + min_dist)) & (df_reg['x'] < (maxx - min_dist)) & \
        (df_reg['y'] > (miny + min_dist)) & (df_reg['y'] < (maxy - min_dist)) & \
        (df_reg['z'] > 0.1)
    
    return df_reg[I].reset_index(drop = True)

def clean_roi(geom, min_area=1e-4, buffer_tol=0.0):
    """
    Remove tiny sliver polygons from an ROI geometry.

    min_area is in CRS units squared.
    If your CRS is lon/lat, this is degrees^2, which is not ideal.
    Prefer projecting to meters before using area thresholds.
    """

    geom = make_valid(geom)

    # Optional topology cleanup
    geom = geom.buffer(buffer_tol).buffer(-buffer_tol)

    if geom.is_empty:
        return geom

    # If single Polygon
    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_area else Polygon()

    # If MultiPolygon, keep only sufficiently large pieces
    if geom.geom_type == "MultiPolygon":
        parts = [p for p in geom.geoms if p.area >= min_area]
        if len(parts) == 0:
            return Polygon()
        elif len(parts) == 1:
            return parts[0]
        else:
            return MultiPolygon(parts)

    # If GeometryCollection, keep polygonal pieces only
    if geom.geom_type == "GeometryCollection":
        parts = [
            g for g in geom.geoms
            if g.geom_type in ["Polygon", "MultiPolygon"] and g.area >= min_area
        ]
        if len(parts) == 0:
            return Polygon()
        return shapely.union_all(parts)

    return geom

def tag_duplicates(df, lat_col="lat", lon_col="lon", time_col="datetime", region_col="region",
                   max_km=30.0, max_sec=2.0):
    """
    Find duplicate events across regions, based on spatial, temporal, and region criteria.
    Returns a copy of df with a new column 'duplicate_cluster_id'.
    
    Duplicate definition:
    - within max_km kilometers
    - within max_sec seconds
    - different region
    """

    df = df.copy().reset_index(drop=True)
    
    # Convert time to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Convert coords to radians for haversine BallTree
    coords = np.deg2rad(df[[lat_col, lon_col]].to_numpy())
    tree = BallTree(coords, metric="haversine")

    # Query neighbors within max_km
    radius = max_km / 6371.0  # km -> radians
    neighbors = tree.query_radius(coords, r=radius)

    # Build graph of duplicates
    G = nx.Graph()
    G.add_nodes_from(df.index)

    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if i >= j:
                continue
            # Check time criterion
            dt = abs((df.loc[i, time_col] - df.loc[j, time_col]).total_seconds())
            if dt > max_sec:
                continue
            # Check region criterion
            if df.loc[i, region_col] == df.loc[j, region_col]:
                continue
            # Add edge = duplicate relationship
            G.add_edge(i, j)

    # Find connected components = duplicate clusters
    clusters = list(nx.connected_components(G))

    # Assign cluster IDs (non-duplicates get NaN)
    cluster_id = {}
    for cid, comp in enumerate(clusters):
        if len(comp) > 1:  # only keep clusters with >1 point
            for idx in comp:
                cluster_id[idx] = cid

    df["duplicate_cluster_id"] = df.index.map(cluster_id).astype("float")
    return df

def longitude_distance(lat_deg, lon1_deg, lon2_deg, radius_km=6371):
    """
    Calculate the east-west distance between two longitudes at a given latitude.

    Parameters:
    lat_deg (float): Latitude in degrees (positive north, negative south).
    lon1_deg (float): First longitude in degrees.
    lon2_deg (float): Second longitude in degrees.
    radius_km (float): Radius of Earth in kilometers (default 6371 km).

    Returns:
    float: Distance in kilometers.
    """
    # Convert to radians
    lat_rad = math.radians(lat_deg)
    lon1_rad = math.radians(lon1_deg)
    lon2_rad = math.radians(lon2_deg)

    # Difference in longitude
    delta_lambda = abs(lon2_rad - lon1_rad)

    # Distance formula
    distance = radius_km * math.cos(lat_rad) * delta_lambda
    return distance

def get_region_poly(_reg):
    geodat = pkl.load(open(os.path.join(polydat, _reg + '_data_fixed.pkl'), 'rb'))
    lon = geodat['coords']['longrid']
    lat = geodat['coords']['latgrid']
    bottom = np.column_stack([lon[0, :],      lat[0, :]])
    right  = np.column_stack([lon[:, -1],     lat[:, -1]])
    top    = np.column_stack([lon[-1, ::-1],  lat[-1, ::-1]])
    left   = np.column_stack([lon[::-1, 0],   lat[::-1, 0]])
    boundary = np.vstack([bottom, right[1:], top[1:], left[1:]])
    return shapely.Polygon(boundary)

def get_overlap(ol_list):
    ol_df = gpd.GeoDataFrame(geometry = ol_list, crs = 'EPSG:4326')
    intsct = reduce(lambda x, y: x.intersection(y), ol_df.geometry)
    
    # contains overlapped regions
    result_gdf = gpd.GeoDataFrame(geometry=[intsct], crs=ol_df.crs)  
    return ol_df, result_gdf

def get_events_in_overlap(ol_list, catmerge, roi):
    ev_in_reg = pd.DataFrame()
    for regpt in ol_list:
        ev_in_reg = pd.concat([ev_in_reg, catmerge[catmerge['region'] == regpt]])
        
    olpoints = []
    for _, x in ev_in_reg.iterrows():
        xgeom = shapely.Point(x['lon'], x['lat'])
        if xgeom.intersection(roi).item():
            olpoints.append(x)
        
    olpoints = gpd.GeoDataFrame(olpoints).reset_index(drop = True).sort_values(['ot']).reset_index(drop = True)
    olpoints['dtime'] = np.hstack([np.array([np.nan]), np.diff(olpoints['ot'])])
    olpoints['datetime'] = [UTC(x).datetime for x in olpoints['ot']]
    return olpoints

def ellipsoid_vol(c_xx, c_xy, c_xz, c_yy, c_yz, c_zz):
    a, b, c, d, e, f = c_xx, c_xy, c_xz, c_yy, c_yz, c_zz
    det = a*d*f + 2*b*c*e - a*e**2 - d*c**2 - f*b**2
    c_rad =  np.sqrt(7.815)  # e.g., chi2(3, p=0.95)
    V = (4/3)*np.pi*(c_rad**3)*np.sqrt(det)
    return V
        

def rm_duplicates(olpoints):
    events_filt = []
    # be sure to automatically keep points with nan duplicate cluster id
    for olid in tqdm.tqdm(list(set(olpoints['duplicate_cluster_id'])), total = len(list(set(olpoints['duplicate_cluster_id'])))):
        idlocs = olpoints[olpoints['duplicate_cluster_id'] == olid].reset_index(drop = True)
        idlocs = idlocs[idlocs['dtime'] > 0].reset_index(drop = True)
        #print(len(idlocs))
        pickdat = []
        for _, xloc in idlocs.iterrows():
            nll_obs_dir = os.path.join('nonlinloc', 
                                   xloc['region'], 
                                   str(xloc['datetime'].date()),
                                   'loc')
            files = glob.glob(nll_obs_dir + '/' + xloc['evid'] + '*' + 'loc.hyp')
            try:
                hypfile = [x for x in files if '.sum.' not in x][0]
            except IndexError:
                continue
            picktxt = open(hypfile, 'r').readlines()[18:-3]
            picks = [x.split('>')[0] for x in picktxt]
            pickdat.append(picks)
            
        # Flatten while keeping unique items per list
        all_items = [item for l in pickdat for item in set(l)]

        # Count occurrences across lists
        counts = Counter(all_items)

        # Strings in 2 or more lists
        common_some = {item for item, c in counts.items() if c > 1}
       
        #print(common_some) 
        
        if len(common_some) > 0:
            # consider a duplicate -- favor the one with the smaller azgap
            event_take = idlocs.sort_values(['gap']).reset_index(drop = True).loc[0]
            events_filt.append(event_take)
        
        elif len(common_some) == 0:
            for _, x in idlocs.iterrows():
                events_filt.append(x)
            
    for _, xev in olpoints[pd.isna(olpoints['duplicate_cluster_id'])].iterrows():
        events_filt.append(xev)
        
    filtdf = pd.DataFrame(events_filt)
    print(len(filtdf), 'events remaining')
    return filtdf
    


regions = ['W1', 'W2', 'W3', 'E1', 'E2', 'E3']

for reg in regions:
    catalog = pd.read_csv('nonlinloc/catalog_ssst_final/' + reg + '_cat.csv')
    dat = pkl.load(open(os.path.join('geographic/crescent_cvm_regions_v2', reg + '_data_fixed.pkl'), 'rb'))
    
    
    known = (dat['coords']['latgrid'].reshape(-1), 
             dat['coords']['longrid'].reshape(-1))
    
    query = (catalog['lat'].to_numpy(), catalog['lon'].to_numpy())
    
    ztopo = griddata(known, dat['topo']['dem'].reshape(-1), query, method = 'linear')
    
    dep = catalog['dep'] - ztopo
    
    catalog['newdep'] = dep
    catalog.to_csv('nonlinloc/catalog_ssst_final/' + reg + '_topocorr.csv')
    
#%%

rt = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia'
catfiles = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/nonlinloc'
polydat = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/geographic/crescent_cvm_regions_v2'

jsfile = json.load(open(os.path.join(rt, 'cascadia_poly.json'), 'r'))
poly_regions = pd.DataFrame(jsfile)

regions = ['W1', 'W2', 'W3', 'E1', 'E2', 'E3']

events_v3 = pd.DataFrame()
for reg in regions:
    file = os.path.join(catfiles, 'catalog_ssst_final' , reg + '_topocorr.csv')
    geodat = pkl.load(open(os.path.join(polydat, reg + '_data_fixed.pkl'), 'rb'))

    df = pd.read_csv(file)
    df['region'] = [reg for x in range(len(df))]
    # first, remove the edge events
    df_clean = remove_edge_events(df, geodat, 3)
    
    df_plot = df_clean[(df_clean['max_err'] < 3) & (df_clean['gap'] < 200)].reset_index(drop = True)
    # fig, ax = plt.subplots()
    # ax.scatter(df_plot['lon'], df_plot['lat'], s = 0.1, c = df_plot['dep'], vmin = 0, vmax = 60)
    # ax.set_title(reg)
    events_v3 = pd.concat([events_v3, df_clean])
    
df_clean2 = events_v3[(events_v3['max_err'] < 5) & (events_v3['gap'] < 250)].reset_index(drop = True)
df_clean2 = df_clean2[df_clean2['lat'] > 39.8]
#df_clean2.to_csv('nonlinloc/catalog_ssst_final/merged_topocorr.csv')



# catfiles = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/nonlinloc'
# polydat = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/geographic/crescent_cvm_regions_v2'

# rt = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia'

# os.chdir(rt)

#catmerge = pd.read_csv(os.path.join(catfiles, 'catalog_ssst_final/merged_topocorr.csv'))
catmerge = df_clean2.copy()
#catmerge['region'] = [x.split('.')[0] for x in catmerge['evid']]

jsfile = json.load(open(os.path.join(rt, 'cascadia_poly.json'), 'r'))
poly_regions = pd.DataFrame(jsfile)

#%%

regions = ['W1', 'W2', 'W3', 'E1', 'E2', 'E3']

quads = {
        'Q1' : ['W1', 'W2', 'E1', 'E2'],
        'Q2' : ['W2', 'W3', 'E2', 'E3']
}

# the regions of intersection and exclusion zones
overlaps = {
        'W1E1' : [['W1', 'E1'], ['Q1']],
        'W1W2' : [['W1', 'W2'], ['Q1']],
        'E1E2' : [['E1', 'E2'], ['Q1']],
        'W2E2' : [['W2', 'E2'], ['Q1', 'Q2']],
        'W2W3' : [['W2', 'W3'], ['Q2']],
        'W3E3' : [['W3', 'E3'], ['Q2']],
        'E2E3' : [['E2', 'E3'], ['Q2']]
}


# all overlap polys
all_polys = []
quad_poly = {}
overlap_roi = []
# iterate over the keys
# then get the polygon for the quadrant of overlap

for key in quads.keys():
    ol_regions = []
    ol = quads[key]
    for reg in ol:
        ol_regions.append(get_region_poly(reg))        
    
    # contains overlapped regions
    ol_df, result_gdf = get_overlap(ol_regions)
    
    roi_geom = result_gdf.loc[0].geometry
    roi_geom = clean_roi(roi_geom, min_area=1e-4)
    
    roi = gpd.GeoSeries([roi_geom], crs=ol_df.crs)
    
    olpoints = get_events_in_overlap(ol, catmerge, roi)
    olpoints = tag_duplicates(olpoints)
    filtdf = rm_duplicates(olpoints)
    
    quad_poly[key] = {
                    'regions' : ol,
                    'polys'   : ol_df,
                    'overlap' : result_gdf
                    }
    overlap_roi.append(roi)
    all_polys.append(filtdf)



for ov in overlaps.keys():
    olreg = overlaps[ov][0]
    
    # get the target region
    overs = []
    for olx in olreg:
        overs.append(get_region_poly(olx))
        
    ol_df, result_gdf = get_overlap(overs)
    oldf0 = gpd.GeoDataFrame(geometry = overs)
    
    
    exdf = pd.DataFrame()
    for qr in overlaps[ov][1]:
        exdf = pd.concat([exdf, quad_poly[qr]['overlap']])
        
    
    if ov == 'W2E2':
        roi0 = result_gdf.difference(gpd.GeoDataFrame(exdf.iloc[[0]]))
        
        roi1 = result_gdf.difference(gpd.GeoDataFrame(exdf.iloc[[1]]))
        
        _, roi2 = get_overlap([roi0.loc[0], roi1.loc[0]])
        
        roi = gpd.GeoSeries(roi2.loc[0])
        
        roi_geom = roi.item()
        roi_geom = clean_roi(roi_geom, min_area=1e-4)
        
        #roi = gpd.GeoSeries([roi_geom], crs='EPSG:4326')
        roi = gpd.GeoSeries([roi_geom], crs=ol_df.crs)
        #roi = roi0.difference(gpd.GeoDataFrame(exdf.iloc[[1]]))
        
       #  fig, ax = plt.subplots()
       # # roi0.plot(ax = ax, facecolor = 'grey', edgecolor = 'k')
       #  exdf.iloc[[0]].plot(ax = ax, edgecolor = 'red')
       #  exdf.iloc[[1]].plot(ax = ax, edgecolor = 'blue')
       #  roi.plot(ax = ax, edgecolor = 'k', facecolor = 'grey')
        
    else:
        roi = gpd.GeoSeries(result_gdf.difference(exdf).loc[0])
        
        roi_geom = roi.item()
        roi_geom = clean_roi(roi_geom, min_area=1e-4)
        
        #roi = gpd.GeoSeries([roi_geom], crs='EPSG:4326')
        roi = gpd.GeoSeries([roi_geom], crs=ol_df.crs)
    
    olpoints = get_events_in_overlap(olreg, catmerge, roi)
    
    if ov == 'W1W2':
        
        msh_exclude = shapely.Polygon(((-122.8, 46.0),
                                      (-121.4, 46.0),
                                      (-121.4, 46.85),
                                      (-122.8, 46.85),
                                      (-122.8, 46.0)))
        # remove all events from this area
        # msh area: assume the W1 region...
        olfilt = []
        msh_keep = []
        for _, x in olpoints.iterrows():
            xgeom = shapely.Point(x['lon'], x['lat'])
            if not xgeom.within(msh_exclude):
                olfilt.append(x)
            elif xgeom.within(msh_exclude):
                if x['region'] == 'W1':
                    msh_keep.append(x)
        
        # overlap_roi.append(msh_exclude)
        # all_polys.append(pd.DataFrame(msh_keep))
        # olpoints = pd.DataFrame(olfilt)
        all_polys.append(pd.DataFrame(msh_keep))
        olpoints = pd.DataFrame(olfilt)
    
    olpoints = tag_duplicates(olpoints)
    filtdf = rm_duplicates(olpoints)
    
    all_polys.append(filtdf)
    overlap_roi.append(gpd.GeoSeries(roi))    
    


all_ol_roi = []
for roii in overlap_roi:
    if type(roii) == gpd.GeoSeries:
        all_ol_roi.append(roii.item())
    else:
        all_ol_roi.append(roii)

all_ol_df = gpd.GeoDataFrame(geometry = all_ol_roi)

non_ol_events = pd.DataFrame()
for regi in regions:
    regcat = catmerge[catmerge['region'] == regi].reset_index(drop = True)
    temp_gdf = gpd.GeoDataFrame(regcat, 
                      geometry = gpd.points_from_xy(regcat['lon'], regcat['lat']),
                      crs = 'EPSG:4326'
                      )
    # pts_keep = []
    # for _, x in regcat.iterrows():
    #     xgeom = shapely.Point(x['lon'], x['lat'])
    #     if not xgeom.within(all_ol_df).any():
    #         pts_keep.append(x)
    joined = gpd.sjoin(temp_gdf, all_ol_df, how="left", predicate="intersects")
    joined_i = joined[pd.isna(joined['index_right'])].index
    non_ol_events = pd.concat([non_ol_events, regcat.iloc[joined_i]])
    
# add all events from overlap regions
#for region in regions:
allevents = pd.DataFrame()
for dff in all_polys:
    try:
        allevents = pd.concat([allevents, dff.drop(['duplicate_cluster_id'])])
    except:
        allevents = pd.concat([allevents, dff])
        
allfilt = pd.concat([allevents, non_ol_events])
#testpoly = pd.DataFrame(all_polys)
allfilt.to_csv('nonlinloc/catalog_ssst_final/filt_events_duplicates_removed.csv')

allfilt = pd.read_csv('nonlinloc/catalog_ssst_final/filt_events_duplicates_removed.csv')
#%%

testplot = allfilt[(allfilt['max_err'] < 5) & (allfilt['gap'] < 250)].reset_index(drop = True)
fig, ax = plt.subplots()
cb = ax.scatter(testplot['lon'], testplot['lat'], s = 0.05, c = testplot['z'], vmin = 0, vmax = 60)
fig.colorbar(cb)
fig.savefig('nonlinloc/catalog_ssst_final/duplicates_removed_v2.png', dpi = 300)