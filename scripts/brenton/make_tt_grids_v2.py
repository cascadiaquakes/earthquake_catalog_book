#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:20:09 2025

make the travel time grids for the mount st helens stations--for testing

@author: bhirao
"""
import gc
import pykonal
import numpy as np
import time
import pymap3d as pm
import pickle as pkl
import os
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
import math
from pyproj import Geod
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import matplotlib.pyplot as plt
import nllgrid
import argparse


def dx_dy_geodesic(lat1, lon1, lat2, lon2):
    """
    Get dx (east) and dy (north) in meters between two geographic points
    using geodesic calculations on the WGS84 ellipsoid.
    """
    geod = Geod(ellps="WGS84")
    
    # Forward and back azimuths, and distance
    fwd_azimuth, back_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)
    
    # Convert azimuth to radians
    az_rad = math.radians(fwd_azimuth)
    
    # Resolve into components
    dx = distance * math.sin(az_rad)  # Eastward
    dy = distance * math.cos(az_rad)  # Northward
    
    return dx, dy


def nearest_grid_index(point_xyz, grid_x, grid_y, grid_z):
    """
    Find the index (i,j,k) of the nearest grid point to a given 3D point.

    Parameters:
        point_xyz: tuple/list/array of (x, y, z) for the point
        grid_x, grid_y, grid_z: 3D arrays (same shape) with coordinates of each grid point

    Returns:
        i, j, k: indices of nearest grid point
    """
    px, py, pz = point_xyz[0], point_xyz[1], point_xyz[2]

    # Compute squared distance to every grid point
    dist_sq = (grid_x - px)**2 + (grid_y - py)**2 + (grid_z - pz)**2
    
    dist_sq_2 = np.sqrt(dist_sq)

    # Find the index of the smallest distance
    flat_index = np.argmin(dist_sq)

    # Convert back to (i,j,k)
    return np.unravel_index(flat_index, grid_x.shape), dist_sq_2[np.unravel_index(flat_index, grid_x.shape)]

def fill_nearest_3d(grid):
    """
    Fill NaN/Inf values in a 3D grid using nearest-neighbor interpolation.

    Parameters
    ----------
    grid : np.ndarray
        3D numpy array with NaN or Inf values.

    Returns
    -------
    grid_filled : np.ndarray
        3D array with NaN/Inf replaced by nearest valid values.
    """
    if grid.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")

    grid = grid.astype(float)  # make sure it's float (for NaN support)

    # Mask of valid values
    mask = (np.isfinite(grid)) & (grid < 999)

    # If all entries are valid, return a copy
    if mask.all():
        return grid.copy()

    # Find nearest valid value for each voxel
    nearest = distance_transform_edt(~mask,
                                     return_distances=False,
                                     return_indices=True)

    return grid[tuple(nearest)]


#def make_hdr_text():
    

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-grid", "--grid", help = 'which grid?', type = str)
parser.add_argument("-job", "--job", help = 'Which job [0-100]', type = int)
args = parser.parse_args()

job = args.job
reg = args.grid
#%%

NJOBS = 20
MINDAYS = 365

grids = 'geographic/crescent_cvm_regions_v2'

outfile_root = 'geographic/crescent_cvm_simulps'
project_root = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia'
os.chdir(project_root)

vgdir = 'nonlinloc/vgrid'
#reg = 'W1'
dat = pkl.load(open(os.path.join(grids, reg + '_data_fixed.pkl'), 'rb'))

nx, ny, nz = dat['coords']['shape'][0], dat['coords']['shape'][1], dat['coords']['shape'][2]
dx, dy, dz = 1, 1, 1

print(f'Grid Range: in x direction {(nx-1)*dx} km, in y direction {(ny-1)*dy} km, in z direction {(nz-1)*dz} km')

# load or build up velocity
p_vel_structure = dat['vp3d']

# set up stations 
stations = pd.read_csv(os.path.join(project_root, 'geographic', 'grid_stas', 'stas_' + reg + '.csv'))
stations = stations.drop_duplicates([' Station '])
stations = stations[stations['cont_days'] > MINDAYS].reset_index(drop = True)
stations['job_id'] = np.tile(np.arange(NJOBS), len(stations) // NJOBS + 1)[:len(stations)]

run_stas = stations[stations['job_id'] == job].reset_index(drop = True)

if not os.path.exists('nonlinloc/ttgrid/' + reg):
    os.makedirs('nonlinloc/ttgrid/' + reg, exist_ok = True)

#msh_stas = '/projects/amt/shared/cascadia_data_mining/MSH/tm_catalog/mshstations.txt'
#mshdf = pd.read_csv(msh_stas, sep = '|')
#%%
# check for nan or infinite values in tt grid
#for _,stadat in tqdm(run_stas.iterrows(), total = len(run_stas)):
  #  stadat = stations[stations[' Station '] == 'VALT']
for _,stadat in tqdm(run_stas.iterrows(), total = len(run_stas)):
    p_time_table = np.zeros_like(dat['vp3d'])
    
    #srclat, srclon, srcdep = stadat[' Latitude '], stadat[' Longitude '], -stadat[' Elevation ']/1E3
    srclat, srclon, srcdep = stadat[' Latitude '], stadat[' Longitude '], 0.
    
    lat0 = dat['coords']['lat0']
    lon0 = dat['coords']['lon0']
    
    gx, gy = dx_dy_geodesic(lat0, lon0, srclat, srclon)
    
    mg = np.meshgrid(dat['coords']['yvals'][:ny], dat['coords']['xvals'][:nx], dat['coords']['zvals'][:nz])
    
    src_idx, dnear = nearest_grid_index((gy/1E3,gx/1E3,srcdep), *mg)
    vp_at_sta = dat['vp3d'][src_idx]
    tt_known = dnear/vp_at_sta
    
    # get distance to this point, then just 
    
    #gps2dist_azimuth(lat0, lon0, srclat, srclon)
    
    #time0 = time.time()
    #    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver = pykonal.solver.PointSourceSolver(coord_sys = 'cartesian')
    #solver.velocity.min_coords = dat['coords']['xvals'].min()/1E3, dat['coords']['yvals'].min()/1E3, dat['coords']['zvals'].min()
    solver.velocity.min_coords = dat['coords']['xvals'].min(), dat['coords']['yvals'].min(), dat['coords']['zvals'].min()
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = dat['coords']['shape']
    solver.velocity.values = p_vel_structure            
    
    solver.traveltime.values[src_idx] = tt_known
    #solver.src_loc = pykonal.transformations.geo2sph(np.array([srclat, srclon, srcdep]))
    solver.src_loc = pykonal.transformations.geo2sph(np.array([gx/1E3, gy/1E3, srcdep]))
    
    solver.unknown[src_idx] = False
    #solver.trial.push(*src_idx)
    solver.solve()
    #solver.trace_ray()
    tp = solver.traveltime.values
    tp_fixed = fill_nearest_3d(tp)
   # tp_fixed = np.swapaxes(tp_fixed, 0,1)
    
    tp_nll = nllgrid.NLLGrid()
    
    tp_nll.array = tp_fixed
    tp_nll.dx, tp_nll.dy, tp_nll.dz = 1., 1., 1.    
    
    tp_nll.x_orig = dat['coords']['xvals'].min()
    tp_nll.y_orig = dat['coords']['yvals'].min()
    tp_nll.z_orig = 0
    tp_nll.type = 'TIME'
    tp_nll.origin_lat = lat0
    tp_nll.origin_lon = lon0
    tp_nll.orig_lat = lat0
    tp_nll.orig_lon = lon0
    tp_nll.proj_name = 'SIMPLE'
   #tp_nll.proj_ellipsoid = 'WGS-84'
   # tp_nll.first_std_paral = dat['latgrid'].min() + 0.5
   # tp_nll.second_std_paral = dat['latgrid'].max() - 0.5
    tp_nll.station = stadat[' Station ']
    tp_nll.sta_x = gx/1E3,
    tp_nll.sta_y = gy/1E3,
    tp_nll.sta_z = 0
    
    # '{} {:.6f} {:.6f} {:.6f}\n'.format(
        # str(tp_nll.station), tp_nll.sta_x, tp_nll.sta_y, tp_nll.sta_z)
    
    # '%s %.6f %.6f %6f' % (tp_nll.station, tp_nll.sta_x, tp_nll.sta_y, tp_nll.sta_z)
    
   # fn = os.path.join('nonlinloc/ttgrid', 'W1', 'ttgrid.P.' + stadat[' Station '] + '.time')
    fn = os.path.join(project_root, 'nonlinloc/ttgrid', reg, 'ttgrid.P.' + stadat[' Station '] + '.time')
    tp_nll.basename = fn
    tp_nll.write_buf_file()
    print(reg, stadat)
  #  tp_nll.write_hdr_file()
    
    #tp_nll.plot()    

    # tp_nll = nllgrid.NLLGrid(basename = 'nonlinloc/ttgrid/W1/ttgrid.P.'+stadat[' Station '] + '.time',
    #                 nx = tp_fixed.shape[0],
    #                 ny = tp_fixed.shape[1],
    #                 nz = tp_fixed.shape[2],
    #                 x_orig = dat['coords']['xvals'].min()/1E3,
    #                 y_orig = dat['coords']['yvals'].min()/1E3,
    #                 z_orig = dat['coords']['zvals'].min(),
    #                 dx = 1.,
    #                 dy = 1.,
    #                 dz = 1)
    
    hdr1 = '%i %i %i %.2f %.2f %.2f %.2f %.2f %.2f TIME\n' %\
             (nx, 
              ny, 
              nz, 
              dat['coords']['xvals'].min(),
              dat['coords']['yvals'].min(),
              0,
              dx,
              dy,
              dz)
            
    hdr2 = '%s %.6f %.6f %.6f \n' %\
             (stadat[' Station '],
              gx/1E3,
              gy/1E3,
              0)
             
    hdr3 = 'TRANSFORM AZIMUTHAL_EQUIDIST WGS-84 LatOrig %.6f LongOrig %.6f RotCW 0.0' %\
            (lat0,
             lon0)
                
    # hdr3 = 'TRANSFORM  TRANS_MERC RefEllipsoid WGS-84 LatOrig %.2f LongOrig %.2f RotCW 0.0 FLOAT' %\
    #         (lat0,
    #           lon0)
            
    # hdr3 = 'TRANSFORM SIMPLE LatOrig %.6f LongOrig %.6f RotCW 0.0' %\
    #         (lat0,
    #           lon0)
    
   # hdr3 = 'TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig 47.500000  LongOrig -122.250000  FirstStdParal 46.250000  SecondStdParal 48.782543  RotCW 0.000000'
    
    open(fn + '.hdr', 'w').writelines([hdr1, hdr2, hdr3])
    

#%%
# for i in range(0, len(sources)):
#     st = time.time()
#     solver = pykonal.EikonalSolver(coord_sys="cartesian")
#     solver.velocity.min_coords = 0, 0, 0
#     solver.velocity.node_intervals = dx, dy, dz
#     solver.velocity.npts = nx, ny, nz
#     solver.velocity.values = p_vel_structure
#     src_idx = int(sources[i,0]/dx), int(sources[i,1]/dy), int(sources[i,2]/dz+40)
#     solver.traveltime.values[src_idx] = 0
#     solver.unknown[src_idx] = False
#     solver.trial.push(*src_idx)
#     for point in air:
#         solver.known[tuple(point[:3].astype('int'))] = True
# #        solver.traveltime.values[tuple(point)] = 999
#     solver.solve()
#     tp = solver.traveltime.values[:,:,0:40].copy()
#     del solver
#     gc.collect()
#     solver = pykonal.EikonalSolver(coord_sys="cartesian")
#     solver.velocity.min_coords = 0, 0, 0
#     solver.velocity.node_intervals = dx, dy, dz
#     solver.velocity.npts = nx, ny, nz
#     solver.velocity.values = s_vel_structure
#     src_idx = int(sources[i,0]/dx), int(sources[i,1]/dy), int(sources[i,2]/dz+40)
#     solver.traveltime.values[src_idx] = 0
#     solver.unknown[src_idx] = False
#     solver.trial.push(*src_idx)
#     for point in air:
#         solver.known[tuple(point[:3].astype('int'))] = True
# #        solver.traveltime.values[tuple(point)] = 999
#     solver.solve()
#     ts = solver.traveltime.values[:,:, 0:40].copy()
#     print(rank,'consumes',time.time()-st)
#     for j in range(0, len(stations)):
#         p_time_table[i][j] = tp[int(stations[j][0]/dx), int(stations[j][1]/dy), 41-int(stations[j][2]/dz)]
#         s_time_table[i][j] = ts[int(stations[j][0]/dx), int(stations[j][1]/dy), 41-int(stations[j][2]/dz)]

# comm.barrier()
# #mpi gather time table
# recv_p = None
# recv_s = None
# if rank == 0:
#     recv_p = np.empty([size, len(sources), len(o_stations)])
#     recv_s = np.empty([size, len(sources), len(o_stations)])
# comm.Gather(p_time_table, recv_p, root=0)
# comm.Gather(s_time_table, recv_s, root=0)
# if rank == 0:
#     np.save('./tt_P',np.vstack(recv_p))
#     np.save('./tt_S',np.vstack(recv_s))
