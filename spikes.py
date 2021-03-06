#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
spiking analysis tools for the CGID datasets
'''

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import cgid.lfp
import cgid.data_loader
import cgid.tools
import numpy as np
from   matplotlib.mlab import find
from   warnings import warn
from   neurotools.stats.modefind  import modefind
from   neurotools.signal.signal   import box_filter
from   neurotools.jobs.decorator  import memoize
from   neurotools.tools           import dowarn
from   cgid.unitinfo import allunitsbysession,classification_results
from   matplotlib.cbook import flatten

def get_all_units(session,area):
    '''
    Reports all sorted units (even if they are noise or multiunit)
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    spikeTimes = np.squeeze(etaloadvariable(session,area,'unitIds'))
    return range(1,len(spikeTimes)+1)

@memoize
def get_spikes_session(session,area,unit,Fs=1000):
    '''
    spikeTimes:
	    1xNUNITS cell array of spike times in seconds. These are raw spike times
	    for the whole session and have not been segmented into individual trials.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    
    Returns
    -------
    '''
    if dowarn(): print('NOTE UNIT  IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    spikeTimes = cgid.data_loader.metaloadvariable(session,area,'spikeTimes')
    return np.int32(spikeTimes[0,unit-1][:,0]*Fs)

@memoize
def get_spikes_session_time(session,area,unit,start,stop):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    start : float
        Start time
    stop : float
        Stop time
    
    Returns
    -------
    '''
    assert stop>start
    assert start>=0
    spikes = get_spikes_session(session,area,unit)
    return spikes[(spikes>=start)&(spikes<stop)]

@memoize
def get_spikes_session_filtered_by_epoch(session,area,unit,epoch):
    '''
    spike times from session.
    spike times outside of trials, from bad trials, and outside of epoch
    on good trials, are removed
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    allspikes = []
    event,est,esp = epoch
    for tr in cgid.data_loader.get_good_trials(session):
        t     = cgid.data_loader.get_trial_event(session,area,tr,4)
        te    = cgid.data_loader.get_trial_event(session,area,tr,event)
        start = t+te+est
        stop  = t+te+esp
        sp = get_spikes_session_time(session,area,unit,start,stop)
        allspikes.append(sp)
    return np.array(list(flatten(allspikes)),dtype=np.int32)

@memoize
def get_spikes_session_raster(session,area,unit,start,stop,decimate=1):
    '''
    Get spikes as raster for a fixed time window relative to session time
    sample rate is 1KHz and sample times are in ms (equiv. samples)
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    start : float
        Start time
    stop : float
        Stop time
    decimate : int, default 1
        If decimate is > 1, spike raster will be down-sampled by decimate
    
    Returns
    -------
    '''
    assert stop>start
    assert start>=0
    spikes = get_spikes_session_time(session,area,unit,start,stop)
    raster = np.zeros((stop-start)//decimate)
    raster[(spikes-start)//decimate]=1
    return raster

@memoize
def get_spikes(session,area,unit,trial):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    if dowarn(): print('NOTE UNIT  IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    if dowarn(): print('NOTE TRIAL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    if dowarn(): print('NOTE EVENT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    assert trial>0
    ByTrialSpikesMS = cgid.data_loader.metaloadvariable(session,area,ByTrialSpikesMS)
    return ByTrialSpikesMS[unit-1,trial-1]

@memoize
def get_spikes_event(session,area,unit,trial,event,start,stop):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    event : int
        Which trial epoch to examine
    start : float
        Start time
    stop : float
        Stop time
    
    Returns
    -------
    '''
    if dowarn(): print('NOTE UNIT  IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    if dowarn(): print('NOTE TRIAL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    if dowarn(): print('NOTE EVENT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    assert trial>0
    ByTrialSpikesMS = cgid.data_loader.metaloadvariable(session,area,'ByTrialSpikesMS')
    spikems = ByTrialSpikesMS[unit-1,trial-1]
    te      = cgid.data_loader.get_trial_event(session,area,trial,event)
    start  += te
    stop   += te
    if len(spikems)==0: return np.array(spikems)
    spikems = spikems[spikems>=start]
    spikems = spikems[spikems<stop]
    return np.unique(np.array(spikems)-start)

def get_spikes_epoch(session,area,unit,trial,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    if type(epoch)==str:
        epoch = epoch.lower()[:3]
        if epoch=='obj':
            return get_spikes_event(session,area,unit,trial,6,-1000,0)
        if epoch=='gri' or epoch=='grp':
            return get_spikes_event(session,area,unit,trial,6,0,1000)
        if epoch=='pla' or epoch=='pre':
            return get_spikes_event(session,area,unit,trial,8,-2000,0)
        if epoch=='mov':
            return get_spikes_event(session,area,unit,trial,8,0,500)
    elif type(epoch)==tuple:
        assert len(epoch)==3
        return get_spikes_event(session,area,unit,trial,*epoch)
    assert 0

def get_spikes_epoch_all_trials(session,area,unit,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    trials = cgid.data_loader.get_good_trials(session)
    allspikes = []
    for trial in trials:
        allspikes.append(get_spikes_epoch(session,area,unit,trial,epoch))
    return np.array(allspikes)

def get_good_units(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    return np.array(sorted(list(allunitsbysession[session,area])))

def get_good_high_rate_units(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    units = get_good_units(session,area)
    ok = set([u for (s,a,u) in cgid.unitinfo.acceptable if (s,a)==(session,area)])
    units = set(units) & ok
    return np.array(sorted(list(units)))

def get_cgid_units_class(session,area,unitclass='Periodic'):
    '''
    Classes are
    Rhythmic
    Burst
    Poisson
    Other
    Created November 2015
    See the file Figure_1_ISIs_and_classifications.py
    All Units have at least 200 ISI events between the first second
    and 1s pre-go periods.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    classifiedunits = {'Burst': {('RUS120518','M1' ): {1},
      ('RUS120518','PMd'): {36,59,73,98},
      ('RUS120518','PMv'): {38,62,90,101,141},
      ('RUS120521','M1' ): {35,57,81,102,118},
      ('RUS120521','PMd'): {26,45,51,93},
      ('RUS120521','PMv'): {23,56,57,87},
      ('RUS120523','M1' ): {1,6,17,59,63,92,106},
      ('RUS120523','PMd'): {4,23,55,97,114},
      ('RUS120523','PMv'): {22,46,47,58,59,92,148,149,157},
      ('SPK120918','M1' ): set(),
      ('SPK120918','PMd'): {4,23,55,84,88,111},
      ('SPK120918','PMv'): {27,37,48,50,51,53,101,114,116,118,127,135,199,205,208},
      ('SPK120924','M1' ): {16,26,110,112},
      ('SPK120924','PMd'): {20,35,53},
      ('SPK120924','PMv'): {31,36,39,42,51,102,108,110,114,185,193},
      ('SPK120925','M1' ): {21},
      ('SPK120925','PMd'): {14,20,35,41,86},
      ('SPK120925','PMv'): {30,62,103,108,133,189,190,195}},
     'Other': {('RUS120518','M1' ): set(),
      ('RUS120518','PMv'): set(),
      ('RUS120521','PMd'): {27},
      ('RUS120521','PMv'): set(),
      ('RUS120523','M1' ): {43},
      ('RUS120523','PMd'): {39},
      ('SPK120918','M1' ): {23},
      ('SPK120918','PMd'): {119},
      ('SPK120918','PMv'): {221},
      ('SPK120924','M1' ): set(),
      ('SPK120924','PMd'): {25,115},
      ('SPK120925','PMd'): {106}},
     'Poisson': {('RUS120518','M1' ): set(),
      ('RUS120518','PMd'): {95},
      ('RUS120518','PMv'): set(),
      ('RUS120521','M1' ): {51},
      ('RUS120521','PMd'): set(),
      ('RUS120521','PMv'): {4},
      ('RUS120523','M1' ): {76,98},
      ('RUS120523','PMd'): {36,60,75},
      ('RUS120523','PMv'): {199},
      ('SPK120918','M1' ): set(),
      ('SPK120918','PMd'): set(),
      ('SPK120918','PMv'): {1,212,222,223},
      ('SPK120924','M1' ): {40,43,54,80},
      ('SPK120924','PMd'): {90,100},
      ('SPK120924','PMv'): {2,74,147,199},
      ('SPK120925','M1' ): {37},
      ('SPK120925','PMd'): set(),
      ('SPK120925','PMv'): {17,201,210,212}},
     'Rhythmic': {('RUS120518','M1' ): {65,66,68,75,88,94,106,107,110},
      ('RUS120518','PMd'): {21,22,30,38,68,69,72,75,77,79,86,89,90,91,93},
      ('RUS120518','PMv'): {15,16,46,49,50,94},
      ('RUS120521','M1' ): {11,19,24,26,31,37,62,64,66,73,86,91,93,96,97,101,105,111},
      ('RUS120521','PMd'): {13,25,31,42,59,62,67,85,98,99},
      ('RUS120521','PMv'): {34,35,81,185},
      ('RUS120523','M1' ): {5,28,38,71,74,81,87,101,104,114,118,119,123,124,128,129,130,133},
      ('RUS120523','PMd'): {1,11,12,22,34,54,56,59,62,65,71,92,96,99,108,117,118,119},
      ('RUS120523','PMv'): {5,36,37,86,94,118,123,139},
      ('SPK120918','M1' ): {2,19,21,52,59,65,80,82,83,86,93,96,118},
      ('SPK120918','PMd'): {12,22,32,35,36,42,43,50,62,70,72,83,87,98,108,109,122},
      ('SPK120918','PMv'): {19,38,63,84,150,165,168,171,174,184,185,187,188,193,198,236},
      ('SPK120924','M1' ): {4,6,18,25,30,35,47,49,53,56,61,62,82,83,87,90,92,93,100,104,107},
      ('SPK120924','PMd'): {3,7,8,11,16,19,23,33,41,42,45,48,59,69,72,75,81,87,91,92,96,97,99,102,104,105,107,112,114,124},
      ('SPK120924','PMv'): {1,5,26,50,87,91,116,120,140,141,159,162,166,175,177,184,188,189,220,221},
      ('SPK120925','M1' ): {10,13,14,15,17,19,22,27,30,46,48,51,52,56,58,59,73,74,77,80,81,84,85,86,98,105,109},
      ('SPK120925','PMd'): {3,7,8,11,12,19,22,30,33,34,43,45,46,50,51,57,62,64,67,68,73,85,91,98,103,104,107,112,113,115,116,123},
      ('SPK120925','PMv'): {1,22,25,26,35,52,54,82,83,85,86,89,94,140,160,163,166,167,171,172,178,179,181,191,220}}}
    return classifiedunits[unitclass][session,area]

def get_channel_ids(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    warn('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    return cgid.data_loader.metaloadvariable(session,area,'channelIds')[0]

def get_channel_id(session,area,unit):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    
    Returns
    -------
    '''
    warn('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    warn('NOTE UNIT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    return get_channel_ids(session,area)[unit-1]

get_unit_channel = get_channel_id
get_channel = get_channel_id

def get_unit_ids(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    return cgid.data_loader.metaloadvariable(session,area,'unitIds')[0]

def get_unit_quality(session,area,unit=None):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    data = cgid.data_loader.metaloadvariable(session,area,'unitQuality')[:,0]
    if unit is None:
        return data
    else:
        return data[unit-1]

def get_good_units_on_channel(session,area,ch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    warn('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    warn('NOTE UNIT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    good = get_good_units(session,area)
    cids = get_channel_ids(session,area)[good-1]
    onthis = find(cids==ch)
    return onthis

def get_all_units_on_channel(session,area,ch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    warn('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    warn('NOTE UNIT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    warn('NOTE THESE ARE ALL UNITS INCLUDING BAD ONES')
    quality = get_unit_quality(session,area)
    use    = find(quality>0)+1
    cids   = get_channel_ids(session,area)[use-1]
    onthis = find(cids==ch)
    return onthis

def get_all_units(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    '''
    warn('NOTE UNIT IS 1 INDEXED FOR MATLAB COMPATIBILITY CONVENTIONS')
    warn('NOTE THESE ARE ALL UNITS INCLUDING BAD ONES')
    quality = get_unit_quality(session,area)
    use = find(quality>0)+1
    return use

def get_spikes_raster(session,area,unit,trial,epoch=None):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    if epoch is None: 
        epoch = (6,-1000,6000)
    eventID,start,stop = epoch
    spikes = get_spikes_epoch(session,area,unit,trial,epoch)
    binned = np.zeros((stop-start,),dtype=int32)
    if len(spikes)>0:
        binned[spikes]=1
    return binned

def get_spikes_raster_all_trials(session,area,unit,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    trials = cgid.data_loader.get_good_trials(session)
    allspikes = []
    for trial in trials:
        allspikes.append(get_spikes_raster(session,area,unit,trial,epoch))
    return np.array(allspikes)

def get_spike_times_all_trials(session,area,unit,epoch,good=True):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    trials = cgid.data_loader.get_good_trials(session) if good \
        else get_valid_trials(session)
    alltrials = []
    for trial in trials:
        alltrials.append(get_spikes_epoch(session,area,unit,trial,epoch))
    return alltrials

def get_all_good_trial_spike_times(session,area,unit,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    return get_spike_times_all_trials(session,area,unit,epoch,good=True)

@memoize
def get_all_good_spike_times(session,area,tr,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    spikes = [get_spikes_epoch(session,area,u,tr,epoch) for u in get_good_units(session,area)]
    return spikes

@memoize
def get_all_good_spike_rasters(session,area,tr,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    spikes = [get_spikes_raster(session,area,u,tr,epoch) for u in get_good_units(session,area)]
    return spikes

@memoize
def get_MUA_spikes(session,area,tr,epoch,ch=None,fsmooth=None,Fs=1000):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    if ch is None:
        use = get_good_units(session,area)
    else:
        use = get_good_units_on_channel(session,area,ch)
    if len(use)>0:
        spikes = np.array([get_spikes_raster(session,area,u,tr,epoch) for u in use])
        mua = np.mean(spikes,0)*Fs
        if not fsmooth is None:
            mua = bandfilter(mua,fb=fsmooth)
    else:
        mua = np.zeros((sp-st,),dtype=float32)
    return mua

get_good_MUA_spikes = get_MUA_spikes
get_MUA_good_spikes = get_MUA_spikes

@memoize
def get_all_MUA_spikes(s,a,tr,epoch,fsmooth=5):
    '''
    
    Parameters
    ----------
    s : string
        Which experimental session to use, for example "SPK120924"
    a : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    return np.array([get_MUA_spikes(s,a,tr,epoch,ch,fsmooth) for ch in get_available_channels(s,a)])


@memoize
def get_all_MUA_spikes_all_areas(session,trial,epoch,fsmooth=5):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    allMUA = []
    for area in areas:
        for u in get_good_units(session,area):
            allMUA.append(get_spikes_raster(session,area,u,trial,epoch))
    MUA = np.array(allMUA)*Fs
    if not fsmooth is None:
        MUA = np.array([bandfilter(x,fb=fsmooth) for x in MUA])
    return MUA


@memoize
def get_all_MUA_spikes_all_areas_average(session,trial,epoch,fsmooth=5,Fs=1000):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    allMUA = []
    for area in areas:
        for u in get_good_units(session,area):
            allMUA.append(get_spikes_raster(session,area,u,trial,epoch))
    MUA = np.mean(allMUA,0)*Fs
    if not fsmooth is None:
        MUA = bandfilter(MUA,fb=fsmooth)
    return MUA


@memoize
def get_MUA_all_spikes(session,area,tr,ch=None,fsmooth=5,epoch=None):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to example. Trial indexing start at 1 for Matlab
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    assert 0 # don't use it
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    if ch is None:
        use = get_all_units(session,area)
    else:
        use = get_all_units_on_channel(session,area,ch)
    if len(use)>0:
        spikes = np.array([get_spikes_raster(session,area,u,tr,epoch) for u in use])
        mua = sum(spikes,0)
        mua = bandfilter(mua,fb=fsmooth)
    else:
        mua = np.zeros((sp-st,),dtype=float32)
    return mua


@memoize
def get_all_MUA_all_spikes(s,a,tr,fsmooth=5,epoch=None):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    assert 0 # don't use it
    return np.array([get_MUA_all_spikes(s,a,tr,ch,fsmooth,epoch) for ch in get_available_channels(s,a)])


@memoize
def get_all_good_spike_times_all_areas(session,trial,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    allspikes = []
    for area in areas:
        spikes = get_all_good_spike_times(session,area,trial,epoch)
        allspikes.extend(spikes)
    return allspikes


@memoize
def get_all_good_spike_rasters_all_areas(session,trial,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    allspikes = []
    for area in areas:
        spikes = get_all_good_spike_rasters(session,area,trial,epoch)
        allspikes.extend(spikes)
    return arr(allspikes)

def spikejitter(x,jitter=50):
    '''
    accepts and returns raster.
    spikes might be jittered outside of range of x, in which case they
    are removed.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    t = find(x)
    t = t+(int32(rand(len(t))*(2*jitter+1))-jitter)
    copy = np.zeros(shape(x),dtype=int32)
    t = t[(t>=0) & (t<len(x))]
    copy[t]=1
    return copy


@memoize
def get_spikes_for_all_units_all_areas(s,epoch,B=None,use=None,decimate=1):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if epoch is None: epoch = 6,-1000,6000 # whole trial
    e,t0,t1 = epoch
    trials = cgid.data_loader.get_good_trials(s)
    allspikes = defaultdict(list)
    for a in areas:
        for tr in trials:
            t = cgid.data_loader.get_trial_event(s,a,tr,4)+\
                cgid.data_loader.get_trial_event(s,a,tr,e)
            for u in get_good_units(s,a) if use is None else use[a]:
                raster = get_spikes_session_raster(s,a,u,t+t0,t+t1,
                    decimate=decimate)
                allspikes[a,u].append(raster)
    return allspikes

@memoize
def get_unit_SNR(session,area,unit):
    '''
    Defined as in Vargas-Irwin and Donoghue 2007
    SNR = 0.5 * (peak-trough) / np.std( first time-bin )
    
    Parameters
    ----------
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    
    Returns
    -------
    '''
    waveforms = cgid.data_loader.get_waveforms(session,area,unit)
    noise     = np.std(waveforms[0,:])
    signal    = 0.5*(np.mean(map(np.max,waveforms))-np.mean(map(np.min,waveforms)))
    return signal/noise

@memoize
def get_mean_waveform(s,a,u):
    '''
    
    Parameters
    ----------
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    
    Returns
    -------
    '''
    wfs = cgid.data_loader.get_waveforms(s,a,u)
    return np.mean(wfs,1)

@memoize
def get_spikes_and_lfp_all_trials(session,area,unit,epoch):
    '''
    
    Parameters
    ----------
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    spikes = get_spikes_epoch_all_trials(session,area,unit,epoch)
    ch     = get_unit_channel(session,area,unit)
    lfps   = np.array([cgid.lfp.get_raw_lfp(session,area,tr,ch,epoch) for tr in cgid.data_loader.get_good_trials(session)])
    return spikes, lfps


@memoize
def unit_class_summary(group,verbose=True):
    '''
    Extracts summary statistics for a tagged group of neurons.
    This is hard coded to use unit categories defined at
    /home/mrule/Desktop/Workspace2/CGID_unit_classification/20141106 manually sort isi wf acorr isolated units/
    >>>> useable,thin,thick,missing = unit_class_summary(group,verbose=True)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    thinthick = classification_results()
    manual_class_dir = '/home/mrule/Desktop/Workspace2/CGID_unit_classification/20141106 manually sort isi wf acorr isolated units/'
    # Units have been manually sorted, locate them based on which directory they were sorted into
    found = cgid.tools.find_all_extension(manual_class_dir+group)
    foundunits = []
    # Extract information about units from the file names of the manually sorted directories
    for u in found:
        try:
            session,area,number,channel,uid = u.split('_')
            number = int(number.split('(')[1][:-1])
            foundunits.append((session,area,number))
        except:
            print('found file',u,'is not a unit summary figure.')
    if verbose:
        print('')
        print(len(foundunits),'units categorized as',group,'(not all may be useable)')
    useable = []
    #print('Breakdown by session, area, and monkey')
    for s,a in cgid.tools.sessions_areas():
        thisarea = [u for (_s,_a,u) in foundunits if (s,a)==(_s,_a)]
        good = set(thisarea)# & set([u for (_s,_a,u) in acceptable if (s,a)==(_s,_a)])
        thin  = [u for u in good if (s,a,u) in thinthick and thinthick[s,a,u]==0]
        thick = [u for u in good if (s,a,u) in thinthick and thinthick[s,a,u]==1]
        miss  = [u for u in good if (s,a,u) not in thinthick]
        #print('\t',s,a,'\t',)
        #print(len(thisarea),'total',)
        #print(len(good),'useable',)
        #if len(good)>0:
        #    print('%d (%d%%)'%(len(thin),len(thin)*100./len(good)),'thin spike',)
        #    print('%d (%d%%)'%(len(thick),len(thick)*100./len(good)),'thick spike',)
        #if len(miss)>0:
        #    print('%d'%len(miss),'missing from thick-thin, check',)
        #print('')
        useable.extend([(s,a,u) for u in good])
    missing = []
    thin  = []
    thick = []
    for s,a,u in useable:
        if (s,a,u) in thinthick:
            if thinthick[s,a,u]==1:
                thick.append((s,a,u))
            elif thinthick[s,a,u]==0:
                thin.append((s,a,u))
            else:
                assert 0
        else:
            missing.append((s,a,u))
    return useable,thin,thick,missing

def poisson_KS(allisi):
    '''
    KS statistics for an ISI distribution against the Poisson (Exponential)
    input: isi events
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NMAX = max(1000,np.max(allisi)*2)
    K = float(len(allisi))
    CDF = cumsum(np.array([sum(allisi==n) for n in range(NMAX)])/K)
    ll = 1./np.mean(allisi)
    CDF2 = 1.0-exp(-ll*arange(NMAX))
    difference = CDF-CDF2
    KS = np.max(abs(difference))
    return KS


def remove_bursts(spikes, duration=5):
    '''
    remove spikes too close together
    input: list of lists of spike times
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    results = []
    for train in spikes:
        if len(train)<1:
            results.append([])
            continue
        previous = train[0]
        newtrain = [previous]
        for spike in train[1:]:
            delta = spike-previous
            if delta<duration:
                pass
                # remove this spike
            else:
                # use this spike
                newtrain.append(spike)
            previous = spike
        results.append(newtrain)
    return np.array(results)


def amap(f,x):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return array(list(map(f,x)))

def get_isi_stats(spikes,epoch,
    FS=1000,BURST_CUTOFF_MS=10,MIN_NISI=100):
    '''
    Computes a statistical summary of an ISI distribution.
    Accepts a list of lists of spike times
    return burstiness, ISI_cv, mean_rate, KS, mode, burst_free_ISI_cv, burst_free_mean_rate, burst_free_mode
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    event,start,stop = epoch
    duration = (stop-start)/float(FS)
    ISI_events = list(flatten(map(diff,spikes)))
    allisi = np.array(ISI_events)
    if len(ISI_events)<MIN_NISI: return None
    mode = FS/modefind(allisi)
    mean_rate = (sum(map(len,spikes))) / float(len(spikes)) / duration
    ISI_cv = np.std(allisi)/np.mean(allisi)
    KS = poisson_KS(allisi)
    burstiness = sum(allisi<BURST_CUTOFF_MS)/float(len(allisi))*100
    burst_free_spikes     = remove_bursts(spikes, duration=BURST_CUTOFF_MS)
    burst_free_ISI_events = list(flatten(map(diff,burst_free_spikes)))
    burst_free_allisi     = np.array(burst_free_ISI_events)
    burst_free_mode       = FS/modefind(burst_free_allisi)
    burst_free_mean_rate  = (sum(
        amap(len,burst_free_spikes))) / float(len(burst_free_spikes)) / duration
    burst_free_ISI_cv     = np.std(burst_free_allisi)/np.mean(burst_free_allisi)
    return burstiness, ISI_cv, mean_rate, KS, mode, burst_free_ISI_cv, burst_free_mean_rate, burst_free_mode

@memoize
def get_isi_stats_unit_epoch(session,area,unit,epoch,MIN_NISI):
    '''
    Computes a statistical summary of an ISI distribution for given unit for all good trials for given epoch

    Returns
    -------
    burstiness, ISI_cv, mean_rate, KS, mode, burst_free_ISI_cv, burst_free_mean_rate, burst_free_mode 
    
    Parameters
    ----------
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    '''
    event,start,stop = epoch
    spikes = []
    for trial in cgid.data_loader.get_good_trials(session):
        spikes.append(cgid.spikes.get_spikes_event(session,area,unit,trial,event,start,stop))
    return get_isi_stats(spikes,epoch,MIN_NISI=MIN_NISI)

@memoize
def get_unfiltered_mua(units,epoch):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    spikes = np.array([cgid.spikes.get_spikes_raster_all_trials(s,a,u,epoch) for (s,a,u) in units])
    return np.mean(spikes,0)

@memoize
def get_boxfiltered_mua(units,epoch,box=25):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    MUA    = np.array([box_filter(tr,25) for tr in get_unfiltered_mua(units,epoch)])
    return MUA


@memoize
def get_epoch_firing_rate(session,area,unit,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    Fs=1000
    spikes = get_spikes_session_filtered_by_epoch(session,area,unit,epoch)
    n_total_spikes     = len(spikes)
    n_total_times      = len(cgid.data_loader.get_good_trials(session))*(epoch[-1]-epoch[-2])
    rate = float(n_total_spikes*Fs)/n_total_times
    return rate

@memoize
def locate_multiunit_hash_on_channel(session,area,channel):
    '''
    For a given session, area, and channel, determines which unit
    ID (1-indexed) corresponds to the multi-unit hash on that channel.
    Note that this may not always be available. Note that some
    "sorted" units may also be multi-unit
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    garbage1 = get_unitQualities(session,area)==0
    garbage2 = get_unitIds(session,area)==0
    assert all(garbage1.T==garbage2)
    # note: channelIds are 1-indexed
    thischannel = cgid.data_loader.get_channelIds(session,area)==channel
    # we use 1-indexing for units, so an offset is added
    multiunit_hash = find(thischannel.T & garbage1)+1
    assert len(multiunit_hash)==1
    unit = multiunit_hash[0]
    return unit

@memoize
def locate_multiunits_on_channel(session,area,channel):
    '''
    Any channel may have one or more units marked as multi-unit.
    This code locates them.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    ch : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    
    Returns
    -------
    '''
    garbage1 = get_unitQualities(session,area)==0
    garbage2 = get_unitIds(session,area)==0
    assert all(garbage1.T==garbage2)
    # These are also multi-unit, but not in the "hash" group
    garbage3 = get_unitQualities(session,area)==1
    # note: channelIds are 1-indexed
    thischannel = cgid.data_loader.get_channelIds(session,area)==channel
    # we use 1-indexing for units, so an offset is added
    units = find( thischannel.T & (garbage1 | garbage3) )+1
    return units

@memoize
def get_hash_on_channel(session,area,trial,channel,epoch):
    '''
    Returns the multi-unit hash (if available) on the specified channel
    Returns a 1ms binned count
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    channel : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    unit = locate_multiunit_hash_on_channel(session,area,channel)
    return get_spikes_raster(session,area,unit,trial,epoch)

@memoize
def get_multiunit_on_channel(session,area,trial,channel,epoch):
    '''
    Returns the multi-unit hash (if available) on the specified channel
    Returns a 1ms binned count
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    channel : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    units = locate_multiunits_on_channel(session,area,channel)
    return sum([get_spikes_raster(session,area,unit,trial,epoch) for unit in units],0)

@memoize
def get_all_spikes_on_channel(session,area,trial,channel,epoch):
    '''
    Returns ALL threshold crossing, including noise and multi-unit
    has as well as potentially well isolated units, on a given
    channel.
    Returns a 1ms binned count
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    channel : int
        Which channel to examine. Channel indexing starts at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    # we use 1-indexing for units, so an offset is added
    units = find(cgid.data_loader.get_channelIds(session,area)==channel) + 1
    if len(units)<1:
        e,a,b = epoch
        return np.zeros(b-a)
    return sum([get_spikes_raster(session,area,unit,trial,epoch) for unit in units],0)

@memoize
def get_all_other_spikes_on_channel(session,area,unit,trial,epoch):
    '''
    Returns ALL threshold crossing, including noise and multi-unit
    has as well as potentially well isolated units, on a given
    channel. EXCLUDING the unit passed as argument.
    Returns a 1ms binned count
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    # we use 1-indexing for units, so an offset is added
    channel = get_channel(session,area,unit)
    units   = find(cgid.data_loader.get_channelIds(session,area)==channel) + 1
    units   = units[units!=unit]
    if len(units)>0:
        return sum([get_spikes_raster(session,area,u,trial,epoch) for u in units],0)
    # Special case, we need to fake it!
    return 0*get_spikes_raster(session,area,unit,trial,epoch)

@memoize
def get_neighbor_MUA(session,area,unit,trial,epoch):
    '''
    Version 0.2
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : int
        Which unit to examine. Unit indexing starts at 1 for Matlab 
        compatibility
    trial : int
        Which trial to example. Trial indexing start at 1 for Matlab
        compatibility
    epoch : int
        Which trial epoch to examine
    
    Returns
    -------
    '''
    channel = get_channel(session,area,unit)
    nn = cgid.tools.neighbors(session,area,False)[channel]
    return sum([get_all_spikes_on_channel(session,area,trial,channel,epoch) for channel in nn],0)
