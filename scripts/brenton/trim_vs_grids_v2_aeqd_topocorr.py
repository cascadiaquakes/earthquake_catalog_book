#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:25:10 2025

cut out the vs grids for each region

trim the vs grid, and define the station list, convert to vp, make the 1d models

implements the topographic correction method applied in stingray, 
where the velocity models are interpolated below the topographic surface

other fixes: minimum value of the p wave model is 2 km/s

upper 5 km is extrapolated using a univariate spline between 5 to 20 km 

considers topography, uses the same topogrpahic correction method as in the MSH paper

citation for topographic data (SRTM plus)

Tozer, B, Sandwell, D. T., Smith, W. H. F., Olson, C., Beale, J. R., &amp; Wessel, P. (2019). Global bathymetry and topography at 15 arc sec: SRTM15+. Distributed by OpenTopography. https://doi.org/10.5069/G92R3PT9. Accessed 2025-09-16

later: use the 1d models produced here and station lists to make the pyocto vmods 
@author: bhirao
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from obspy import UTCDateTime as UTC
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pyocto
from obspy.geodetics import gps2dist_azimuth
from pyproj import Transformer, Proj
from scipy.interpolate import griddata
import pickle as pkl
import nllgrid
import sys
sys.path.append('/projects/amt/shared/cascadia_data_mining/MSH/stingray_msh')
import stingray_utils
import pyproj
from scipy.spatial import cKDTree
import rasterio as rio
from scipy import interpolate



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


def make_interp_grid(lat_min, lat_max, lon_min, lon_max, dep_min, dep_max, spacing_km=1):
    # --- Step 1: Define geographic bounds and spacing ---
    #lat_min, lat_max = 44.5, 45.5
    #lon_min, lon_max = -121.0, -119.5
    #spacing_km = 10  # spacing in kilometers
    spacing_m = spacing_km * 1000
    
    # --- Step 2: Define projection centered in bounding box ---
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    proj = Proj(proj='aeqd', lat_0=center_lat, lon_0=center_lon, units='m')
    to_proj = Transformer.from_proj("epsg:4326", proj, always_xy=True)
    to_latlon = Transformer.from_proj(proj, "epsg:4326", always_xy=True)
    
    # --- Step 3: Project the lat/lon bounds to meters ---
    x_min, y_min = to_proj.transform(lon_min, lat_min)
    x_max, y_max = to_proj.transform(lon_max, lat_max)
    
    # --- Step 4: Create evenly spaced grid in projected coordinates ---
    x_vals = np.arange(x_min, x_max + spacing_m, spacing_m)
    y_vals = np.arange(y_min, y_max + spacing_m, spacing_m)
    z_vals = np.arange(dep_min, dep_max, spacing_km)
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing = 'xy')
    
    # --- Step 5: Convert back to lat/lon ---
    lon_grid, lat_grid = to_latlon.transform(xx, yy)
    
    # Optional: flatten
    lat_points = lat_grid.ravel()
    lon_points = lon_grid.ravel()
    dep_points = zz.ravel()
    
    coords = {'xvals' : x_vals,
              'yvals' : y_vals,
              'zvals' : z_vals, 
              'shape' : xx.shape,
              'lat0'  : center_lat,
              'lon0'  : center_lon}
    
    return lat_points, lon_points, dep_points, xx.shape, coords

def project2utm():
    midlon=-122.1940802193146
    midlat=46.199055781445665 
    dx=dy=dz=0.2 # model node spacing
    xoffset=-30.0
    yoffset=-30.0
    maxdep=15.0
    xdist=60.0
    ydist=60.0
    mshproj = pyproj.Proj("+proj=utm +zone=10 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    utmx, utmy = mshproj(midlon, midlat)
    #minlonact, minlatact= minlon, minlat
    _,maxlon=stingray_utils.ll2dxdy((midlat,midlon),90,xdist/2*1000)
    _,minlon=stingray_utils.ll2dxdy((midlat,midlon),270,xdist/2*1000)
    maxlat,_=stingray_utils.ll2dxdy((midlat,midlon),0,ydist/2*1000)
    minlat,_=stingray_utils.ll2dxdy((midlat,midlon),180,ydist/2*1000)


def ll2dxdy_aeqd(proj, lat0, lon0, dx, dy):
    # Starting point
   # lat0, lon0 = 47.0, -120.0
    
    # Define local projection centered on (lon0, lat0)
 #   aeqd = Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")
    
    # Build transformer
    transformer_to = Transformer.from_proj("epsg:4326", proj)
    transformer_back = Transformer.from_proj(proj, "epsg:4326")
    
    # Forward: lat/lon -> local x/y
    x0, y0 = transformer_to.transform(lat0, lon0)
    
    # Apply offsets
    x1, y1 = x0 + dx, y0 + dy
    
    # Back: local x/y -> lat/lon
    lat1, lon1 = transformer_back.transform(x1, y1)

    return lat1, lon1       

def interp_topo(targ_lon, targ_lat, topodata):
    
    min_lat = targ_lat.min() - 50/111.1
    max_lat = targ_lat.max() + 50/111.1
    min_lon = targ_lon.min() - 50/111.1
    max_lon = targ_lon.max() + 50/111.1
    
    topoX, topoY = np.meshgrid(topodata['lon'], topodata['lat'][::-1])
    
   # Ix = (topoX >= min_lon) & (topoX <= max_lon)
   # Iy = (topoY >= min_lat) & (topoY <= max_lat)
    
   # Ikeep = Ix * Iy
    
    mask = (topoX >= min_lon) & (topoX <= max_lon) & (topoY >= min_lat) & (topoY <= max_lat)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    
    dem_trimmed = topodata['dem'][row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1]
    lat_trimmed = topoY[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1]
    lon_trimmed = topoX[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1]
    
    x = lon_trimmed.ravel()
    y = lat_trimmed.ravel()
    z = dem_trimmed.ravel()
    
    nrows, ncols = lon_trimmed.shape
    
    X_coords = np.vstack([x,
                          y]).T
    
    tree = cKDTree(X_coords)
    
    Xi_coords = np.vstack([targ_lon.reshape(-1),
                          targ_lat.reshape(-1)]).T
    
    values = dem_trimmed.reshape(-1)
    
 #   topo_interp = interpolate.griddata(X_coords, values, Xi_coords)
    
    k = 6  # number of nearest neighbors
    distances, indices = tree.query(Xi_coords, k=k)
    
    # Avoid division by zero
    distances[distances == 0] = 1e-10
    
    # Compute inverse distance weights
    weights = 1 / distances
    weights /= weights.sum(axis=1)[:, None]
    
    # Compute weighted average for interpolation
    z_new = np.sum(weights * z[indices], axis=1)
    dem_interpolated = z_new.reshape(targ_lon.shape)
    
    out = {
           'lon' : targ_lon, 
           'lat' : targ_lat, 
           'dem' : dem_interpolated/1E3
           }
    
    return out

def get_topo():
    #topo = rio.open('geographic/output_SRTMGL3.tif')
    topo = rio.open('geographic/output_SRTM15Plus.tif')
    elev = topo.read(1)
    
    transform = topo.transform   # affine transform
    nrows, ncols = elev.shape

    # Create row and column indices
    rows = np.arange(nrows)
    cols = np.arange(ncols)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Convert pixel indices to lon/lat
    lon_grid, lat_grid = rio.transform.xy(transform, row_grid, col_grid)
   # LON = lon_grid.reshape((nrows, ncols))
    #LAT = lat_grid.reshape((nrows, ncols))
    
    return {'dem' : elev, 'lon' : np.unique(lon_grid), 'lat' : np.unique(lat_grid)}

def interp_shallow(dep_col, d1, d2, minvp = 2):
    'interpolate the upper 0 to 3 km or so, using the upper 5-10'
    # Example data
    x = np.arange(d1, d2)
   # dep_col = dep_col.copy()
    
    dep_col[dep_col < minvp] = minvp
    #y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])
    
    # Fit cubic spline
    spline = interpolate.UnivariateSpline(x, dep_col, k=3, s=0)  # k=3 → cubic, s=0 → interpolate exactly
    
    # Evaluate inside and outside the range
    x_fit = np.linspace(0, d1, d1+1)  # extrapolate beyond data
    y_fit = spline(x_fit)
    
    y_fit[y_fit < minvp] = minvp
    
    return y_fit
    
#%%
    
gtsrc_term = lambda x: ' '.join(['GTSRC', 
                                 x[' Station '], 
                                 'LATLON', 
                                 str(x[' Latitude ']), 
                                 str(x[' Longitude ']),
                                 str(x[' Elevation ']/1E3)
                                 ])

regions = ['W1', 'W2', 'W3', 'E1', 'E2', 'E3']
grids = 'geographic/crescent_cvm_regions'

outfile_root = 'geographic/crescent_cvm_simulps'
project_root = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia'

vgdir = 'nonlinloc/vgrid'

vpvs = pd.read_csv('geographic/vpvs_ratios.csv')

vsmod = pd.read_csv('geographic/crescent_cvm_v0/mcmcVs_20250507.all.dat', sep = '\s+')

print(np.max(vsmod['lat']), np.min(vsmod['lat']), np.max(vsmod['lon']), np.min(vsmod['lon']))
vs_maxlat = np.max(vsmod['lat'])
vs_minlat = np.min(vsmod['lat'])
vs_maxlon = np.max(vsmod['lon'])
vs_minlon = np.min(vsmod['lon'])

DX = 1

minyrs = [1,1,1,1,1,1]
vmods = ['p4', 'o4', 'nc', 'e3', 'o4', 'nc']
end_time = UTC(2024,1,1)
start_time = UTC(2002,1,1)

poly = json.load(open('cascadia_poly.json', 'r'))
ls_years = np.arange(2002,2024,1)

depmin = 0
depmax = 100

perm_stas = pd.read_csv('geographic/permanent_stations.txt', sep = '|')
perm_stas = perm_stas[perm_stas['Network ']!='AM'].reset_index(drop = True)
perm_stas['cont_days'] = [(UTC(x[' EndTime']) - UTC(x[' StartTime ']))/86400 for _, x in perm_stas.iterrows()]

# this is a garbage import...
grids = 'geographic/crescent_cvm_regions'
topodat = get_topo()

#perm_stas = perm_stas[perm_stas[' EndTime'] > UTC(2008,1,1)].reset_index(drop = True)
#%%
polygons = {}
for ii, ipoly in tqdm(enumerate(poly), total = len(poly)):
    #
    reg = ipoly['poly_id']
    # don't use these
   # dat = pkl.load(open(os.path.join(grids, reg + '_data.pkl'), 'rb'))
    
    poly_stas = pd.read_csv(os.path.join('geographic/grid_stas',  'stas_' + reg + '.csv' ))
    
    poly_vpvs = vpvs[vpvs['region'] == ipoly['poly_id']]['vpvs'].item()
    MINDAYS = minyrs[ii] * 365
    
    polygons[ipoly['poly_id']] = ipoly
    pll = (ipoly['minlon'], ipoly['minlat'])
    plr = (ipoly['maxlon'], ipoly['minlat'])
    pur = (ipoly['maxlon'], ipoly['maxlat'])
    pul = (ipoly['minlon'], ipoly['maxlat'])
    
    x0 = ipoly['olon']
    y0 = ipoly['olat']
    
    
    Ilat = (vsmod['lat'] >= ipoly['minlat']) & (vsmod['lat'] <= ipoly['maxlat'])
    Ilon = (vsmod['lon'] >= ipoly['minlon']) & (vsmod['lon'] <= ipoly['maxlon'])
    
    vgrid = vsmod[Ilat & Ilon].reset_index(drop = True)
    vgrid['Vp(km/s)'] = vgrid['Vs(km/s)'] * poly_vpvs
    
    lat0 = ipoly['olat']
    lon0 = ipoly['olon']
    dx=dy=dz=1. # model node spacing
    
    regproj = pyproj.Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, units='m')
    # position of lower left corner
    

    xoffset = round(gps2dist_azimuth(lat0, lon0, lat0, ipoly['minlon'])[0]/1E3)
    yoffset = round(gps2dist_azimuth(lat0, lon0, ipoly['minlat'], lon0)[0]/1E3)
    # these are defined form the middle of the grids
    
   # _, minlon = ll2dxdy_aeqd(regproj, lat0, lon0, -xoffset*1000, 0)
    
    _,minlon = stingray_utils.ll2dxdy((lat0, lon0), 270, xoffset*1000)
    _,maxlon = stingray_utils.ll2dxdy((lat0, lon0), 90, xoffset*1000)
    minlat,_ = stingray_utils.ll2dxdy((lat0, lon0), 180, yoffset*1000)
    maxlat,_ = stingray_utils.ll2dxdy((lat0, lon0), 0, yoffset*1000)
    
    maxdep=depmax

    xdist = xoffset*2
    ydist = yoffset*2

    nx = round(xdist//dx)
    ny = round(ydist//dy)
    nz = int(round(np.abs(depmin) + depmax)/dz)
    
    lats=np.zeros(ny)
   # yvals = np.zeros_like(lats)
    lons=np.zeros(nx)
   # xvals = np.zeros_like(lons)
    for ii in range(nx):
      #  _,lons[ii]=stingray_utils.ll2dxdy((lat0,minlon),90,dx*1e3*ii)
        _, lons[ii]=ll2dxdy_aeqd(regproj, lat0, minlon, dx*1E3*ii, 0)
    for ii in range(ny):
       #lats[ii],_=stingray_utils.ll2dxdy((minlat,lon0),0,dy*1e3*ii)
        lats[ii], _ = ll2dxdy_aeqd(regproj, minlat, lon0, 0, dy*1E3*ii)
        
    depths=np.arange(depmin,depmax+dz,dz)[:int(nz)] # this is depth below the surface
    latgrid,longrid,depgrid=np.meshgrid(lats,lons,depths)
    
    points = (vgrid['lon'].values, vgrid['lat'].values, vgrid['depth'].values)

    interplat = np.ravel(latgrid[depgrid==0])
    interplon = np.ravel(longrid[depgrid==0])
    
    
        #INTERP_TOPO:
    topo_grid = interp_topo(longrid[:,:,0], latgrid[:,:,0], topodat)
    
    zgrid = np.stack([topo_grid['dem']]*depgrid.shape[-1], axis = 2)
    
    pkl.dump(topo_grid, open('geographic/crescent_cvm_regions_v2/topo/' + reg + '.pkl', 'wb'))
    
    g2xyz = np.vstack([[topo_grid['lon'].ravel()], 
                       [topo_grid['lat'].ravel()], 
                       [-topo_grid['dem'].ravel()]]).T
    
    np.savetxt('geographic/crescent_cvm_regions_v2/topo/' + reg + '.xyz', g2xyz, 
               fmt = '%.4f %.4f %.4f', delimiter = '  ')
    
    
    # what we will interpolate it to
    outlats=latgrid.copy().reshape(-1)
    outlons=longrid.copy().reshape(-1)
    outdeps=depgrid.copy().reshape(-1)
    newshape = latgrid.shape
    
    newdeps = outdeps - zgrid.reshape(-1)
    
    known = np.array([[interplat], [interplon]]).squeeze().T
    
   # newpoints = (outlons,outlats,outdeps)
    newpoints = (outlons,outlats,newdeps)
    
    pvals=griddata(points, vgrid['Vp(km/s)'].values, newpoints) # p slowness
    
    vp = pvals.reshape(*latgrid.shape)
    
   # xvals = np.arange(0, (nx + 1) * dx, dx)
    xvals = np.arange(-xoffset, -xoffset + (nx)*dx, dx)
  #  yvals = np.arange(0, (ny + 1) * dy, dy)
    yvals = np.arange(-yoffset, -yoffset + (ny)*dy, dy)
    zvals = np.arange(depmin, depmin + (nz * dz), dz)
    
    ygrid, xgrid, zgrid = np.meshgrid(yvals, xvals, zvals)

    grid_coords = {
                    'xvals' : xvals,
                    'yvals' : yvals,
                    'zvals' : zvals,
                    'shape' : newshape,
                    'lat0' : lat0,
                    'lon0' : lon0,
                    'latgrid' : interplat.reshape(*newshape[:2]),
                    'longrid' : interplon.reshape(*newshape[:2]),
                    'xgrid'   : xgrid[:,:,0],
                    'ygrid'   : ygrid[:,:,0],
                    'zgrid'   : zgrid[:,:,0]
                    }
    
    
    # save the interpolated vp grid, also make the 1d model here
    
    vp_avg = np.zeros(len(np.unique(depths)))
    for ii, dep_i in enumerate(np.unique(depths)):
        depslice = vp[:,:,ii]
        depslice = depslice[depslice != 0.]
        vp_avg[ii] = np.nanmean(depslice)
    
    #inspect the grid lat/lon
    
    # you may want to chop off any values above 0 km bsl, or replace with a surface vp value
    vp1d = pd.DataFrame({'depth(km)' : depths,
                         'Vp'        : vp_avg})
    
    vp1d.to_csv('geographic/crescent_cvm_regions/' + ipoly['poly_id'] + '_1d_v2.csv')
    
    
    POLY = Polygon([pll, plr, pur, pul])
    poly_stas = []
    for _,xsta in perm_stas.iterrows():
        stapoint = Point(xsta[' Longitude '], xsta[' Latitude '])
        if POLY.contains(stapoint):
            poly_stas.append(xsta)
    
    poly_df = pd.DataFrame(poly_stas).reset_index(drop = True)
    poly_df = poly_df[poly_df[' StartTime '] <= (end_time - MINDAYS*86400)]
    poly_df = poly_df[poly_df[' EndTime'] >= (start_time + MINDAYS*86400)]
    
    poly_df = poly_df.drop_duplicates(['Network ', ' Station ']).reset_index(drop = True)
    
    poly_df.to_csv('geographic/grid_stas_cvm/stas_' + ipoly['poly_id'] + '.csv')
    
    
    out_data = {
                'vp'    : vp1d,
                'coords' : grid_coords,
                'vp3d'  : vp,
                'longrid' : longrid,
                'latgrid' : latgrid,
                'depths' : zvals,
                'topo' : topo_grid
                }
    
    pkl.dump(out_data, open('geographic/crescent_cvm_regions_v2/' + ipoly['poly_id'] + '_data.pkl', 'wb'))
    
   # grid3d = dat['vp3d']
    grid3d = vp
    minvel = 2.
    
    xg = xvals
    yg = yvals
    zg = zvals
    
    Ygrid, Xgrid, Zgrid = np.meshgrid(yg, xg, zg, indexing='xy')
    
    Ygrid = Ygrid[:,:,0]
    Xgrid = Xgrid[:,:,0]
    Zgrid = Zgrid[:,:,0]
    
    longrid = longrid[:,:,0]
    latgrid = latgrid[:,:,0]
    
    all_vp = pd.DataFrame()
    vp_export = np.zeros_like(grid3d)
    
    rows, cols = np.indices(longrid.shape[:2])
    row_flat = rows.ravel()
    col_flat = cols.ravel()
    
    for idep, zd in enumerate(zvals[:nz]):
        
       # depslc = grid3d[:,:, idep]
        
        
        if np.all(~np.isnan(grid3d[:,:,idep])):
         #   print('none are nan')
            # if none are nan
            grid_df = pd.DataFrame({'lat' : latgrid.reshape(-1),
                                    'lon' : longrid.reshape(-1),
                                    'x' : Xgrid.reshape(-1),
                                    'y' : Ygrid.reshape(-1),
                                    'z' : np.ones(len(latgrid.reshape(-1))) * zd,
                                    'vp' : grid3d[:,:, idep].reshape(-1),
                                    'row_i' : row_flat,
                                    'col_i' : col_flat
                                    })
            fixed_df = grid_df
            
        elif np.all(np.isnan(grid3d[:,:,idep])):
          #  print('all are nan')
            # if all are nan
            # find the first depth slice that does not have all nan
            grid_df = pd.DataFrame({'lat' : latgrid.reshape(-1),
                                    'lon' : longrid.reshape(-1),
                                    'x' : Xgrid.reshape(-1),
                                    'y' : Ygrid.reshape(-1),
                                    'z' : np.ones(len(latgrid.reshape(-1))) * zd,
                                    'vp' : np.ones(len(latgrid.reshape(-1))) * minvel,
                                    'row_i' : row_flat,
                                    'col_i' : col_flat                                    
                                    })      
            
            fixed_df = grid_df
            
        else:
            # if some are nan
          #  print('some nan')
            grid_df = pd.DataFrame({'lat' : latgrid.reshape(-1),
                                    'lon' : longrid.reshape(-1),
                                    'x' : Xgrid.reshape(-1),
                                    'y' : Ygrid.reshape(-1),
                                    'z' : np.ones(len(latgrid.reshape(-1))) * zd,
                                    'vp' : grid3d[:,:, idep].reshape(-1),
                                    'row_i' : row_flat,
                                    'col_i' : col_flat                                    
                                    })    
            
            fixed_df = fill_nan_with_nearest_neighbors(grid_df, ['vp'])
       
        all_vp = pd.concat([all_vp, fixed_df])   
        vp_export[:, :, idep] = fixed_df['vp'].to_numpy().reshape(nx, ny)
        
        # upper 2 km has a bunch of weird stuff...
        # consider extrapolating from 3-10 km
        # consider cutting off the bottom 10 km
   
    for ixx in range(vp_export.shape[0]):
        for iyy in range(vp_export.shape[1]):
            dep_col = vp_export[ixx, iyy, 5:20]
            shallow_col = interp_shallow(dep_col, 5, 20)
            vp_export[ixx, iyy, :len(shallow_col)] = shallow_col
            
            
    # remove this section
    # for ik in range(vp_export.shape[0]):
    #     for jk in range(vp_export.shape[1]):
    #        # column = vp_export[ik,jk,:]
    #         elev_node = topo_grid['dem'][ik,jk]
    #         elev_ind = int(np.ceil(elev_node) + 3)
    #         inds = np.arange(0,4,1)
    #         inds = inds[(inds) < elev_ind][:-2]
    #         vp_export[ik,jk,inds] = 0.343
            
   # vp_df = pd.DataFrame(all_vp)
        ### Write Vs Grid
    dat_out = out_data.copy()
    dat_out['vp3d'] = vp_export
    dat_out['topo'] = topo_grid
    pkl.dump(dat_out, open(os.path.join(grids + '_v2', reg + '_data_fixed.pkl'), 'wb'))
    
    fig, ax = plt.subplots()
    cb = ax.pcolormesh(longrid, latgrid, vp_export[:,:,0])
    fig.colorbar(cb)
    ax.set_title(reg + ' top of vm')
    fig.savefig('geographic/' + reg + 'vmtop.png', dpi = 300, bbox_inches = 'tight' )
    
    
    fig, ax = plt.subplots()
    cb = ax.pcolormesh(longrid, latgrid, topo_grid['dem'])
    fig.colorbar(cb)
    ax.set_title(reg + ' topo')
    fig.savefig('geographic/' + reg + '_topo.png', dpi = 300, bbox_inches = 'tight' )

    # nlloc_args = {
    #             'reg'       : reg,
    #             'proj_root' : project_root,
    #             'polydata'     : ipoly,
    #             'stations'     : poly_stas,
    #             'runfile_out'  : 'nonlinloc/vgrid/' + '.'.join([reg, 'g2t.run'])
    #             }
    
   # make_nlloc_run(dat, nlloc_args)
    # vp_nll = nllgrid.NLLGrid()
    
    # vp_nll.array = vp_interp_grid
    # vp_nll.dx, vp_nll.dy, vp_nll.dz = 1., 1., 1.    
    
    # vp_nll.x_orig = dat['coords']['yvals'].min()/1E3
    # vp_nll.y_orig = dat['coords']['xvals'].min()/1E3
    # vp_nll.z_orig = dat['coords']['zvals'].min()
    
    
    
    # vp_nll.type = 'TIME'
    # vp_nll.origin_lat = lat0
    # vp_nll.origin_lon = lon0
    # vp_nll.orig_lat = lat0
    # vp_nll.orig_lon = lon0
    # vp_nll.proj_name = 'LAMBERT'
    # vp_nll.proj_ellipsoid = 'WGS-84'
    # vp_nll.first_std_paral = dat['latgrid'].min() + 0.5
    # vp_nll.second_std_paral = dat['latgrid'].max() - 0.5
    # vp_nll.station = stadat[' Station ']
    # vp_nll.sta_x = gx/1E3,
    # vp_nll.sta_y = gy/1E3,
    # vp_nll.sta_z = -stadat[' Elevation ']/1E3
    
    #net_sta = ['.'.join([x['Network '], x[' Station ']]) for _,x in poly_df.iterrows()]
    #poly_df['net.sta'] = net_sta
    
  #  polygons[ipoly['poly_id']]['stas'] = poly_df
  #  polygons[ipoly['poly_id']]['nets'] = list(set(poly_df['Network ']))   
    
    # max_dist = gps2dist_azimuth(ipoly['minlat'], ipoly['minlon'], 
    #                              ipoly['maxlat'], ipoly['maxlon'])[0]/1000
    
 #   mod_path = 'geographic/pyocto_vmod_cvm/' + ipoly['poly_id'] + '_pyocto.vmod'
    
  #  for _, ista in poly_df.iterrows():
    #pyocto.VelocityModel1D.create_model(poly_vm, DX, max_dist + 30, 250, mod_path)
