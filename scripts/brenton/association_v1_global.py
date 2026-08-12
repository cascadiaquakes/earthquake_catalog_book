#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:11:50 2026

include global picks for W3 and E3 grids

E1/2 2014, not working?

take away the pnsn name referencing...

for W3 grid, you can look at the 2022 December 20 ferndale earthquake

performance at the MTJ is fine,
but very poor at the Geysers...
- probably something to do with short S2sta distances
- s waves are poorly identified
- maybe also the geophone network

@author: bhirao
"""
import sys
import pyocto
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
import pickle as pkl
import json
import math
import numpy as np
sys.path.append('/gpfs/projects/amt/shared/cascadia_data_mining/MSH/directory_tools')
import os_tools
from obspy import Stream
import tqdm
import argparse
from obspy import UTCDateTime as UTC
from obspy.clients.fdsn.client import Client
import random

def get_pnsn_events():
    dat = pd.read_csv('benchmark_data/ml_enhanced_catalog.csv')
   # dat = dat.drop_duplicates([''])
    return dat

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

def plot_waveform(evplot, assignments, evstas, clients):
    
    chan_keep = ['HHZ', 'HHE', 'HHN', 
                 'HNZ', 'HNE', 'HNN'
                 'EHZ', 'EHE', 'EHN', 
                 'BHZ', 'BHE', 'BHN', 
                 'EH1', 'EH2','EHZ',
                 'DPZ', 'DPE', 'DPN']
    yoff_inc = 2
    LEN = 50
    xplot = np.arange(0,LEN,1/100)
    colors = ['tab:blue', 'tab:orange']

   # for _, evplot in events.iterrows():
    fig, ax = plt.subplots(figsize = (15,15))
    yoff = 0
    evpicks = assignments[assignments['event_idx'] == evplot['idx']].reset_index(drop = True)
    pickstas = evpicks['station'].to_list()
    evstas = pd.DataFrame([x for _,x in grid_stas.iterrows() if x['sta_id'] in pickstas])
    evstas['dist'] = [haversine_distance(evplot['latitude'], evplot['longitude'], x[' Latitude '], x[' Longitude ']) for _,x in evstas.iterrows()]
    evstas = evstas.sort_values(['dist']).reset_index(drop = True)
    for ista, sta in evstas.iterrows():
        
        if sta['datacenter'] == 'iris':
            client_op = clients['IRIS']
        elif sta['datacenter'] == 'ncedc':
            client_op = clients['NCEDC']
            
        netplot = sta['Network ']
        staplot = sta[' Station ']
        sta_picks = evpicks[evpicks['station'] == sta['sta_id']].reset_index(drop = True)
        loc = sta_picks['loc'].reset_index(drop = True)[0]
        chantype = sta_picks['chantype'].reset_index(drop = True)[0]
        
        chantype = chantype + '*'
        
        if netplot == 'BG':
            chantype = 'DP*'
        
        if loc == '':
            loc = '*'
        
        try:
            wf = os_tools.get_waveform(sta['Network '], sta[' Station '], evplot['utc'].replace(hour = 0, minute = 0, second = 0, microsecond = 0))
        except FileNotFoundError:
            try:
                wf = client_op.get_waveforms(netplot, staplot, loc, chantype, 
                                                  evplot['utc'], evplot['utc'] + LEN)
            except:
                continue
        wf = Stream([x for x in wf if x.stats.channel in chan_keep])
        wf.trim(evplot['utc'], evplot['utc'] + LEN)
        wf.detrend('linear')
        wf.interpolate(100)
        wf.taper(0.05)
        wf.filter('highpass', freq = 1.)
        for itr, tr in enumerate(wf):
            yplot = tr.data/np.max(np.abs(tr.data)) - yoff
            try:
                ax.plot(xplot, yplot[:LEN*100], color = colors[ista%2], linewidth = 0.4)
            except ValueError:
                ax.plot(xplot[:yplot.shape[0]], yplot, color = colors[ista%2], linewidth = 0.4)
            label = '.'.join([tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel])
            if itr == 0:
                ax.text(xplot[-1], -yoff, label, ha = 'right')
            else:
                ax.text(xplot[-1], -yoff, tr.stats.channel, ha = 'right')
            for _, ipick in sta_picks.iterrows():
                pick_x = (ipick['time']-evplot['utc'])
                phase = ipick['phase']
                pick_color = 'black' if phase == 'P' else 'tab:red'
                ax.vlines(pick_x, - yoff - 1, -yoff + 1, color = pick_color)
                
                if itr == 0:
                    ax.text(pick_x, -yoff + 1, phase +  ' %.2f' % ipick['peak_val'])
            yoff += yoff_inc
    
    title = '; '.join([evplot['new_id'], str(evplot['utc'])])
    ax.set_title(title)
    fig.savefig(os.path.join('test_plots_global', evplot['new_id'] + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()
    
#%%    
parser = argparse.ArgumentParser()
parser.add_argument("-grid", "--grid", help = 'which grid?', type = str)
parser.add_argument("-year", "--year", help = 'Which year', type = int)
parser.add_argument("-plots", "--plots", help = 'you want plots?', type = bool)
parser.add_argument("-ibatch", "--ibatch", help = 'which batch?', type = int)
args = parser.parse_args()

grid = args.grid
year = args.year
plots = args.plots 
ibatch = args.ibatch
min_prob = 0.15
NJOBS = 50

#%%

# these you can get from the IRISDMC
pnsn_nets = ['CC', 'UW', 'UO', 'CN', 'GM', 'GS', 'PB', 'C8', 'TA']

# for norcal, need to use NCEDC client
# for Nevada (NN), need SCEDC

#client_nc = Client('NCEDC')
#client_pn = Client('IRIS')

# clients = {'NCEDC' : Client('NCEDC'),
#            'IRIS'  : Client('IRIS')}

dates = pd.date_range(start = datetime.datetime(year,1,1),
                      end = datetime.datetime(year + 1, 1, 1),
                      freq = '1D')

dates = pd.DataFrame({'dates' : dates}, index = np.arange(len(dates)))
dates['ijob'] = dates.index % NJOBS 

rundates = dates[dates['ijob'] == ibatch].reset_index(drop = True)

dates = [x.to_pydatetime() for x in rundates['dates']]

#dates_to_plot = random.sample(dates, 10)

pnsn_df = get_pnsn_events()

#grid = 'W1'
grid_json = json.load(open('geographic/cascadia_poly.json', 'r'))
grid_data = [x for x in grid_json if x['poly_id'] == grid][0]
grid_stas = pd.read_csv('geographic/grid_stas/stas_' + grid + '.csv')
grid_stas['sta_id'] = ['.'.join([x['Network '], x[' Station ']]) + '.' for _, x in grid_stas.iterrows()]

datacenter = ['pnsn' if row['Network '] in pnsn_nets else 'ncedc' for _, row in grid_stas.iterrows()]
grid_stas['datacenter'] = datacenter

grid_vm = pyocto.VelocityModel1D(os.path.join('geographic/pyocto_vmod_cvm/' + grid + '_pyocto.vmod'), tolerance = 1)
grid_assoc = pyocto.OctoAssociator.from_area(
    lat = (float(grid_data['minlat']), float(grid_data['maxlat'])),
    lon = (float(grid_data['minlon']), float(grid_data['maxlon'])),
    zlim = (float(grid_data['zmin']), float(grid_data['zmax'])),
    time_before = 300,
    velocity_model = grid_vm,
    n_picks = 5,
    n_s_picks = 1
)


stalist = grid_stas['sta_id'].to_list()

stations_in = grid_stas[['sta_id', ' Longitude ', ' Latitude ', ' Elevation ']]
stations_in = stations_in.rename({'sta_id' : 'id',
                                  ' Longitude ' : 'longitude',
                                  ' Latitude ' : 'latitude',
                                  ' Elevation ': 'elevation'
                                  }, axis = 1)
stations_in = grid_assoc.transform_stations(stations_in)

# ferndale 2022 event: datetime.datetime(2022,12,20)
# the geysers M5 datetime.datetime(2016,12,14)
# some other event north of the MTJ: datetime.datetime(2012,2,13)
# E3: some event at fort bidwell, NZ datetime.datetime(2014,11,07)
for run_date in tqdm.tqdm(dates, total = len(dates)):

    #test_date = datetime.datetime(2019,7,12)
    #test_date = datetime.datetime(2017,2,23)
    outdir = os.path.join('association', grid, str(year))
    evfname = os.path.join(outdir, str(UTC(run_date).julday) + '.events')
    assfname = os.path.join(outdir, str(UTC(run_date).julday) + '.picks')
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if os.path.exists(evfname) & os.path.exists(assfname):
        continue
        
    picks_root = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/picks'
    picksdir = os.path.join(picks_root, str(run_date.year), 
                            str(run_date.timetuple().tm_yday).zfill(3))
    
    filedf = pd.DataFrame({'filename' : os.listdir(picksdir)})
    filedf['sta_id'] = ['.'.join([x.split('.')[0],x.split('.')[1]]) + '.' for x in filedf['filename']]
    
    pickfiles = pd.DataFrame([x for _,x in filedf.iterrows() if x['sta_id'] in stalist]).reset_index(drop = True)
    if len(pickfiles)==0:
        continue
    stas_have = pickfiles['sta_id'].to_list()
    pickdata2 = []
    for _, pf in pickfiles.iterrows():
        sta_picks = pkl.load(open(os.path.join(picksdir, pf['filename']), 'rb'))
        loc = '*'
        chantype = pf['filename'].split('.')[2]
        for pick in sta_picks:
            dat = {
                    'start_time' : pick.start_time,
                    'end_time' : pick.end_time,
                    'peak_val' : pick.peak_value,
                    'peak_time': pick.peak_time,
                    'phase'    : pick.phase,
                    'trace_id' : pick.trace_id,
                    'loc'      : loc,
                    'chantype' : chantype
                     }
            pickdata2.append(dat)
            
    pickdf = pd.DataFrame(pickdata2)
    pickdf = pickdf[pickdf['peak_val'] >= min_prob].reset_index(drop = True)
    
    # access the global picks
    if grid in ['W3', 'E3']:
        picks_glob = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/picks_global'
        glob_nets = os.listdir(picks_glob)
        stas_missing = [x for x in stalist if x not in stas_have]
        
        stas_to_load = []
        all_glob_picks = pd.DataFrame()
        for sta_get in stas_missing:
            # construct path
            net = sta_get.split('.')[0]
            sta = sta_get.split('.')[1]
            sd1 = os.path.join(picks_glob, net, 
                                  str(run_date.year), 
                                  str(run_date.timetuple().tm_yday).zfill(3)
                                  )
           # sloc = sta_get.split('.')
            # wfday = client.get_waveforms(net, sta, '0*', '*', 
            #                      UTCDateTime(run_date), 
            #                      UTCDateTime(run_date) + 86400)
            
            try:
                fget = [x for x in os.listdir(sd1) if sta + '.' + net in x]
            except FileNotFoundError:
                continue
            
            # for multiple pickfiles of the same station:
            # find the one with the greatest number of picks with prob > 0.3
            
            if len(fget) == 0:
                continue
            
            if len(fget) > 1:
                
                npicks = []
                for ifile, pickfile in enumerate(fget):
                    pkdf = pd.read_csv(os.path.join(sd1, pickfile), compression = 'gzip')
                    pkdf = pkdf[pkdf['confidence'] > min_prob].reset_index(drop = True)
                    npicks.append(len(pkdf))
                    
                    pf_takeI = np.argmax(np.array(npicks))
            
                net = fget[pf_takeI].split('.')[1]
                sta = fget[pf_takeI].split('.')[0]
                chantype = fget[pf_takeI].split('.')[2]
                loc = fget[pf_takeI].split('.')[3]
                
                globdf = pd.read_csv(os.path.join(sd1, fget[pf_takeI]), compression = 'gzip')
                globdat = pd.DataFrame({
                    'start_time' : globdf['peak_time'],
                    'end_time'   : globdf['peak_time'],
                    'peak_val'   : globdf['confidence'],
                    'peak_time'  : [UTCDateTime(x) for x in globdf['peak_time']],
                    'phase'      : globdf['phase'],
                    'trace_id'   : [x['network_code'] + '.' + str(x['station_code']) + '.' for _,x in globdf.iterrows()],
                    'loc'        : [loc] * len(globdf),
                    'chantype'   : [chantype] * len(globdf)
                    })
                globdat = globdat[globdat['peak_val'] >= min_prob]
            
            if len(fget) == 1:
                
                globdf = pd.read_csv(os.path.join(sd1, fget[0]), compression = 'gzip')
                globdat = pd.DataFrame({
                    'start_time' : globdf['peak_time'],
                    'end_time'   : globdf['peak_time'],
                    'peak_val'   : globdf['confidence'],
                    'peak_time'  : [UTCDateTime(x) for x in globdf['peak_time']],
                    'phase'      : globdf['phase'],
                    'trace_id'   : [x['network_code'] + '.' + str(x['station_code']) + '.' for _,x in globdf.iterrows()],
                    'loc'        : [loc] * len(globdf),
                    'chantype'   : [chantype] * len(globdf)
                    })
                globdat = globdat[globdat['peak_val'] >= min_prob]
            
            all_glob_picks = pd.concat([all_glob_picks, globdat])
            
    if grid in ['W1', 'W2', 'E1', 'E2']:        
    ### fix this for W1-2, E1-2
        all_glob_picks = pickdf
    elif grid in ['W3', 'E3']:
        all_glob_picks = pd.concat([all_glob_picks, pickdf])
    
        
    if len(pickdf) == 0:
        print('no picks on ', str(run_date))
        continue
 
    all_glob_picks = all_glob_picks.rename({'trace_id' : 'station',
                                            'peak_time': 'time'}, 
                                           axis = 1)

    time0 = UTCDateTime()
    events, assignments = grid_assoc.associate(all_glob_picks, stations_in)
    events = grid_assoc.transform_events(events)
    print((UTCDateTime() - time0)/60, 'minutes')
    if len(events) == 0:
        print('no events on ', str(run_date))
        continue
    events['utc'] = [UTCDateTime(x) for x in events['time']]
    
    # try:
    #     pnsn_events = Client('IRIS').get_events(
    #                                             starttime = run_date, 
    #                                             endtime = run_date + datetime.timedelta(days = 1),
    #                                             minlatitude = grid_data['minlat'],
    #                                             maxlatitude = grid_data['maxlat'],
    #                                             minlongitude = grid_data['minlon'],
    #                                             maxlongitude = grid_data['maxlon']
    #                                             )
        
    #     pnsn_list = [x for x in pnsn_events]
    # except:
    #     pnsn_list = []
    #     pass



    new_ids = []
    #ev_types = []
    for _, ev in events.iterrows():

        event_id = grid + '.' + str(ev['utc'].year) + str(ev['utc'].month).zfill(2) + \
            str(ev['utc'].day).zfill(2) + str(ev['utc'].hour).zfill(2) + \
                str(ev['utc'].minute).zfill(2) + str(ev['utc'].second).zfill(2)
        #ev_types.append('new')
        new_ids.append(event_id)
            
    events['new_id'] = new_ids
  #  events['type'] = ev_types
    
    events.to_csv(evfname)
    assignments.to_csv(assfname)
    
   # ev2plot = events.sample(10)
    
    # ev2plot = events
    # if plots:
    #     #if run_date in dates_to_plot:
    #     for _, evplot in ev2plot.iterrows():
    #         evpicks = assignments[assignments['event_idx'] == evplot['idx']].reset_index(drop = True)
    #         pickstas = evpicks['station'].to_list()
    #         evstas = pd.DataFrame([x for _,x in grid_stas.iterrows() if x['sta_id'] in pickstas])
            
    #         #try:
    #         plot_waveform(evplot, assignments, evstas, clients)
    #        # except:
           #     continue
    
    #unique_chantypes = list(set(all_glob_picks['chantype']))

