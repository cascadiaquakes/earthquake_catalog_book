#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:17:31 2025

convert the pyocto pickfiles to nonlinloc

also make the runfiles

one obsfile and one runfile for each event





@author: bhirao
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import sys
sys.path.append('/gpfs/projects/amt/shared/cascadia_data_mining/MSH/directory_tools')
sys.path.append('/gpfs/projects/amt/shared/cascadia_data_mining/MSH/nlloc_code')
import nlloc_tools
import pickle as pkl
from obspy import UTCDateTime
import tqdm
import argparse

def make_nlloc_run(dat, evt, evid, obsfile, ttgrid, outdir, vpvs):

    dx, dy, dz = 1, 1, 1
    CONTROL = 'CONTROL 3 12345'
    # TRANS = ' '.join(['TRANS  SIMPLE', 
    #                   str(dat['coords']['lat0']),
    #                   str(dat['coords']['lon0']),
    #                   str(0.0)])
    TRANS = ' '.join(['TRANS  AZIMUTHAL_EQUIDIST WGS-84', 
                      str(dat['coords']['lat0']),
                      str(dat['coords']['lon0']),
                      str(0.0)])
    
    #TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig 47.500000  LongOrig -122.250000  FirstStdParal 46.250000  SecondStdParal 48.782543  RotCW 0.000000
    
 #   TRANS = ' '.join(['TRANS LAMBERT WGS-84',
 #                     str(dat['coords']['lat0']),
 #                     str(dat['coords']['lon0']),
 #                     '46.250000',
 #                     '48.782543',
 #                     '0.0'])
    locfile = outdir + '/' + evid   
 
    LOCSIG = 'LOCSIG B HIRAO'
    LOCCOM = ' '.join(['LOCCOM', str(evt.date()), evid])
    LOCFILES = ' '.join(['LOCFILES', obsfile, 'NLLOC_OBS', ttgrid, locfile])
    LOCHYPOUT = 'LOCHYPOUT SAVE_NLLOC_ALL'
    LOCSEARCH = 'LOCSEARCH OCT 8 8 8 0.01 500000 5000 0 1'
    LGdim = '%i %i %i %.6f %.6f %.6f %.2f %.2f %.2f ' %\
            (dat['coords']['shape'][0], 
              dat['coords']['shape'][1], 
              dat['coords']['shape'][2], 
              dat['coords']['xvals'].min(),
              dat['coords']['yvals'].min(),
              0,
              dx,
              dy,
              dz)
            
    # LGdim = '%i %i %i %s %s %.12f %.2f %.2f %.2f ' %\
    #          (dat['coords']['shape'][0], 
    #           dat['coords']['shape'][1], 
    #           dat['coords']['shape'][2], 
    #           '-213.971164',
    #           '-190.784885',
    #           dat['coords']['zvals'].min(),
    #           dx,
    #           dy,
    #           dz)           
            
#-213.971164 -190.784885            

    LOCGRID = ' '.join(['LOCGRID', 
                        LGdim,
                        'PROB_DENSITY',
                        'SAVE'])

    # LOCGRID = ' '.join(['LOCGRID', 
    #                     LGdim,
    #                     'MISFIT',
    #                     'SAVE'])
    
    LOCMETH = 'LOCMETH EDT_OT_WT 9999.0 4 -1 -1 ' + str(vpvs) + ' -1'
    #LOCMETH = 'LOCMETH GAU_ANALYTIC 9999.0 4 -1 -1 ' + str(vpvs) + ' -1'
    LOCGAU = 'LOCGAU 0.0 0.0'
    LOCPHASEID_P = 'LOCPHASEID  P   P'
    LOCPHASEID_S = 'LOCPHASEID  S   S'
    LOCQUAL = 'LOCQUAL2ERR 0.0'
    LOCMAG = 'LOCMAG ML_HB 1.0 1.110 0.00189'
    LOCPHSTAT = 'LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9'
    NLLargs = '\n'.join([CONTROL, 
                       TRANS, 
                       LOCSIG, 
                       LOCCOM, 
                       LOCFILES, 
                       LOCHYPOUT, 
                       LOCSEARCH,
                       LOCGRID,
                       LOCMETH,
                       LOCGAU,
                       LOCPHASEID_P,
                       LOCPHASEID_S,
                       LOCQUAL,
                       LOCMAG,
                       LOCPHSTAT])
    return NLLargs, locfile

def pick2nlloc_v2(pick_series):
   # ofile = file_out
    stanet = pick_series.station
    
    #pd.concat([ph, filt_picks])
    
    net = stanet.split('.')[0]
    sta = stanet.split('.')[1]
   # net = pick_series.id.split('.')[0]
    phase = pick_series.phase
    time = UTCDateTime(pick_series['time'])
    pick_error = 0.1
    
    pick_txt = (
        "%-6s %-4s %-4s %-1s %-6s %-1s "
        "%04d%02d%02d %02d%02d "
        "%7.4f %-3s %9.2e %9.2e %9.2e %9.2e\n" % \
        (
            sta,
            net,
            "?",
            "?",
            phase,
            "?",
            time.year,
            time.month,
            time.day,
            time.hour,
            time.minute,
            time.second+time.microsecond/1.e6,
            "GAU",
            pick_error,
            -1,
            -1,
            -1,
        )
    )    
    return pick_txt
#%%

parser = argparse.ArgumentParser()
parser.add_argument("-grid", "--grid", help = 'which grid?', type = str)
#parser.add_argument("-job", "--job", help = 'Which job [0-100]', type = int)
args = parser.parse_args()


grids = 'geographic/crescent_cvm_regions_v2'
project_root = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia'

vgdir = 'nonlinloc/vgrid'
reg = args.grid
dat = pkl.load(open(os.path.join(grids, reg + '_data_fixed.pkl'), 'rb'))

vpvs = pd.read_csv('geographic/vpvs_ratios.csv')
vpvs = vpvs[vpvs['region'] == reg]['vpvs'].item()

nx, ny, nz = dat['coords']['shape'][0], dat['coords']['shape'][1], dat['coords']['shape'][2]
dx, dy, dz = 1, 1, 1

rt = '/gpfs/projects/amt/shared/cascadia_data_mining/MSH'
dest_rt = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/nonlinloc/' + reg

ttgrid = '/gpfs/projects/amt/shared/cascadia_data_mining/ML_cascadia/nonlinloc/ttgrid/'+ reg + '/ttgrid'

pyocto_root = os.path.join(project_root, 'association', reg)

t1 = datetime(2002,1,1)
t2 = datetime(2023,12,31)
date_range = pd.date_range(start = t1, end = t2, freq = 'D')
date_range = [x.to_pydatetime() for x in date_range]

runfiles = []
locfiles = []
eventsout = []

#%%
# d = date_range[6401]
for d in tqdm.tqdm(date_range, total = len(date_range)):
    
    sdir = os.path.join(pyocto_root, str(d.year), str(d.timetuple().tm_yday))
    events = sdir + '.events'
    picks = sdir + '.picks'
    
    if (not os.path.exists(events)) or (not os.path.exists(picks)):
        continue

    eventdf = pd.read_csv(events) 
    pickdf = pd.read_csv(picks)

    newdir = os.path.join(dest_rt, str(d.date()), 'obs')
    rundir = os.path.join(dest_rt, str(d.date()), 'run')
    locdir = os.path.join(dest_rt, str(d.date()), 'loc')
    
    if not os.path.exists(newdir):
        os.makedirs(newdir, exist_ok = True)
    if not os.path.exists(rundir):
        os.makedirs(rundir, exist_ok = True)
    if not os.path.exists(locdir):
        os.makedirs(locdir, exist_ok = True)        
        
    for _, event in eventdf.iterrows():
        evpicks = pickdf[pickdf['event_idx'] == event['idx']].reset_index(drop = True)
        obsfile = os.path.join(newdir, event['new_id'] + '.obs')
        
        pickdata = []
        for _, pick in evpicks.iterrows():
            picktxt = pick2nlloc_v2(pick)
            pickdata.append(picktxt)
        obsf = open(obsfile, 'w').writelines(pickdata)
        runfiledat, locdat = make_nlloc_run(dat, 
                                            UTCDateTime(event['utc']).datetime, 
                                            event['new_id'],
                                            obsfile,
                                            ttgrid,
                                            locdir, 
                                            vpvs)
        open(os.path.join(rundir, event['new_id'] + '.run'), 'w').writelines(runfiledat)
        runfiles.append(os.path.join(rundir, event['new_id'] + '.run'))
        locfiles.append(os.path.join(locdir, event['new_id']))
        eventsout.append(event['new_id'])
    
locdf = pd.DataFrame({
                    'event_id'      : eventsout,
                      'locfiles'      : locfiles,
                      'runfile'     : runfiles
                      })
locdf.to_csv(os.path.join('nonlinloc', reg + '_locfiles.csv'))
                
#%% for testing

# outdf = pd.DataFrame({'files' : runfiles})
# outdf.to_csv('nonlinloc', reg + '_test_runfiles.csv')


# for rf in tqdm.tqdm(runfiles[:20], total = len(runfiles[:20])):
#     subprocess.run(['NLLoc', rf])


    # files = [x for x in os.listdir(sdir('usgs_cat2')) if 'uw' in x]
    # for event in files:
    #     obs = os.path.join(sdir(''), 'usgs_cat2',event, 'nlloc3/obs', event + '.obs')
    #     dest_f = os.path.join(dest_rt, str(d.date()), 'obs', event + '.obs')
    #     if os.path.exists(obs):
    #         shutil.copy2(obs, dest_f)
    #         runfiledat, locdat = make_nlloc_run(dat, d, event, dest_f, ttgrid, locdir, vpvs)
    #         open(os.path.join(rundir, event + '.run'), 'w').writelines(runfiledat)
    #         runfiles.append(os.path.join(rundir, event + '.run'))
    #         locfiles.append(locdat)
    #         events.append(event)
    #         imush.append(sdir('usgs_cat2/' + event +'/nlloc3/loc/last.hyp'))

