#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
The documentation for the CGID archive format is important
This archive contains extracted Cued Grasp with Instructed Delay (CGID)
data for session RUS120521 area PMv. The variables are as follows:

README:
    This message

ByTrialLFP1KHz:
    LFP data segmented by trial. This is a NCHANNELx1 cell array. Most areas
    do not have all channels, and only channel numbers that were recorded
    are stored. Each channel is a NTRIALx7000 array. This is seven seconds of
    LFP data at 1KHz starting at 1 second before "ObjectPresent". Note:
    for rusty and spike, the StartTrial events occur about 1 second before
    object presentation. For Gary, they occur about 15ms before object
    presentation. ObjectPresent is the first consistent task event marker
    available for alignment. Trials that did not advance as far as object
    presentation have been discarded.

ByTrialSpikesMS:
    NUNITSxNTRIALS cell array. Each entry is a list of spike times for the
    corresponding cell and trial, in MS, and shifted so that StartTrial
    is time 0

UnsegmentedLFP1KHz:
    This is the original LFP data. It is stored in a 96x1 cell array, where
    the index into the cell array corresponds to the LFP channel number.
    For most arrays, not all channels are available. Missing channels do
    not contiain data. Each channel entry is a NSAMPLESx1 sequence of LFP
    values sampled at 1KHz. These samples have been downsampled from 30KHz
    in two stages. First, a 500Hz 4th order low-pass Butterworth filter is
    used, and the LFPs are resampled at 2KHz. Second, Matlab Decimate is
    used to downsample them to 1KHz, which applies a 400Hz cutoff
    Chebyshev filter.

eventsByTrial:
    NTRIALSx12 trial summary information.
    The 1 column contains a flag indicating which trials are good.
    The 2 column contains a flag indicating object type. 0=KG 1=TC
    The 3 column contains a flag indicating grip type. 0=Pre 1=Pow
    the 4 column contains time (MS) that we start taking data for a trial
    The remaining columns contain trial events, if available, in milliseconds,
        and relative to object presentation. Missing events are denoted with
        -1. Most error trials will have missing events. The event codes, in
        order, are:
        5    StartTrial
        6    ObjectPresent
        7    GripCue
        8    GoCue
        9    StartMov
        10    Contact
        11    BeginLift
        12    EndLift
        13    Reward

spikeTimes:
    1xNUNITS cell array of spike times in seconds. These are raw spike times
    for the whole session and have not been segmented into individual trials.

unitIds:
    1xNUNITS array of unit IDs. A unit can be identified as "unit [unitID] on
    channel [channelID]".

channelIds:
    1xNUNITS array of channel IDs. A unit can be identified as "unit [unitID]
    on channel [channelID]".

waveForms:
    Original waveform data from spike sorting. It is a 1xNUNITS cell array.
    Each entry contains 48xNSPIKES matrix of spiking wavforms for that unit.

availableChannels:
    a 96x1 array of flags. 1 means that a channel is present, 0 means that it
    is not. For Gary and Costello, all even numbered channels were discarded,
    though spikes were collected from all 96 channels. For Rusty and Spike,
    PMv has a full 96 channels. M1 and PMd share a single bank of 96 channels,
    and are hooked up randomly. So a random 50% of M1 and PMd channels are not
    assigned in Rusty and Spike datasets.

arrayChannelMap
    An NxK mapping of the spatial locations of various channels. Positions with
    -1 have no corresponding channels. Matrix is structures so that, when
    displayed in standard format, recording wires exit to the right.

unitQuality
    An NUNITSx1 list of unit quality codes.
    0 : noise ( not a unit )
    1 : poorly isolated ( multiunit or very noisy unit )
    2 : isolated
    3 : well isolated

There are variables containing the original CGID timestamps in seconds.
These are of the form [Object][Grip][Event]. There are two objects, key-grip
object "KG" and tapered-cylinder object "TC". There are two grips, precision
grip "Pre" and power grip "Pow". Each sucessful trial contains the events
StartTrial, ObjectPresent, GripCue, GoCue, StartMov, Contact, BeginLift,
EndLift, and Reward. Trials may end with an error before all events occur.
Error related events include UndefinedGrip, WrongGrip, and Error. The events
ObjectPresent, GripCue, GoCue, StartMov, and Contact from completed trials
only are copied over to events with "COMPLETE" appended to the name.
"""

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
if sys.version_info<(3,):
    from itertools import imap as map

from   warnings import warn
from   matplotlib.mlab   import find
from   scipy.io          import savemat,loadmat
import numpy as np
import neurotools.tools
import neurotools.jobs.cache
import neurotools.jobs.initialize_system_cache
import cgid.tools
import cgid.lfp
import cgid.spikes
from   neurotools.nlab  import memoize
from   neurotools.tools import dowarn,debug
from   matplotlib.cbook import flatten

from cgid.config import sessions, areas, sessionnames

def archive_name(session,area):
    '''
    Converts a (session,area) tuple into a path to load data from disk. 
    At the moment, the CGID_ARCHIVE variable in cgid.config must be set
    by hand for this to work. (TODO)
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    string 
        File path based on `CGID_ARCHIVE` configuration variable
    '''
    archive = '%s_%s.mat'%(session,area)
    return cgid.config.CGID_ARCHIVE+archive

cgid_matfilecache = {}
def metaloaddata(session,area):
    '''
    Loads a matfile from the provided path, caching it in the global dict
    "matfilecache" using path as the lookup key.

    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
        
    Returns
    -------
    dict
        Complete experimental archive as a `dict`. To load a single 
        variable, see `metaloadvariable`.
    '''
    global cgid_matfilecache
    path = archive_name(session,area)
    if path in cgid_matfilecache:
        return cgid_matfilecache[path]
    data = loadmat(path)
    cgid_matfilecache[path]=data
    return data

@neurotools.jobs.initialize_system_cache.unsafe_disk_cache
def metaloadvariable(session,area,variable):
    '''
    Loads a variable from a matfile at the provided path, caching it using
    the disk memoization decorator.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    variable : string
        Which variable to load from the experiment archive
    
    Returns
    -------
    object
        Loaded variable from session archive; The result is cached 
        in RAM explicitly (for sharing amongst multiple processes) if 
        caching has been enabled.
        
    '''
    global cgid_matfilecache
    path = archive_name(session,area)

    # if CGID archive is already loaded grab the variable from memory
    if path in cgid_matfilecache:
        return cgid_matfilecache[path][variable]

    # Otherwise, try to just load the one variable
    return loadmat(path,variable_names=[variable])[variable]

def preload_cgid():
    '''
    Prefetches all CGID archives and commonly used data. Can be used
    to prepare intermediate processing stages for interactive exploration.
    '''
    # It's faster to just load the whole archives into memory
    for session,area in cgid.tools.sessions_areas():
        prefetched = metaloaddata(session,area)
        print('preloaded',session,area)
    # First prefetch individual variables
    variables = '''
        README ByTrialLFP1KHz ByTrialSpikesMS UnsegmentedLFP1KHz
        eventsByTrial spikeTimes unitIds channelIds waveForms
        availableChannels arrayChannelMap unitQuality'''.split()
    # Then, presets per-channel and per-unit stuff
    for session,area in cgid.tools.sessions_areas():
        # prefetch the LFP channels (UnsegmentedLFP1KHz)
        for ch in get_available_channels(session,area):
            prefetched = cgid.lfp.get_raw_lfp_session(session,area,ch)
        for unit in cgid.spikes.get_all_units(session,area):
            # prefetch spike times
            prefetched = cgid.spikes.get_spikes_session(session,area,unit)
            # prefetch waveforms
            prefetched = get_waveforms(session,area,unit)

@memoize
def get_waveforms(session,area,unit=None):
    '''
    Loads waveforms from CGID archive; 
    Uses memoization to speed up subsequent calls.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    unit : None or int, default None
        Which unit to load (indexing starts at 1 for Matlab comatibility)
        If None, then all units are loaded.

    Returns
    -------
    '''
    waveforms = metaloadvariable(session,area,'waveForms')
    if unit is None: 
        return waveforms
    return waveforms[0,unit-1]

def get_unitIds(session,area):
    '''
    Get `unitIds` from CGID archive
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        unitIds variable loaded from file
    '''
    return metaloadvariable(session,area,'unitIds')[0]

def get_channelIds(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        Channel ID number (starting at 1) for all units in the session
    '''
    return metaloadvariable(session,area,'channelIds')

def get_unitQualities(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        unitQuality: manual categorization, 0 for noise, 1 for multiunit,
        2 for acceptable single-unit, and 3 for well-isolated.
    '''
    return metaloadvariable(session,area,'unitQuality')[:,0]

def readme(session,area):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    string
        The readme string from the matlab archive
    '''
    return metaloadvariable(session,area,'README')[0]

def get_array_map(session,area,removebad=True):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        Array channel map; Only set up for subjects SPK and RUS at the
        moment (some channels needed to be hard coded into the source
        code). 
    '''
    # there is a problem where bad channels were removed from this
    # array before saving in the archive. If you really want all channes
    # we need the original map -- which i've hard coded here
    if 'SPK' in session and area=='PMv' and removebad==False:
       return np.array([[88,  2,  1,  3,  4,  6,  8, 10, 14, 92],
                     [65, 66, 33, 34,  7,  9, 11, 12, 16, 18],
                     [67, 68, 35, 36,  5, 17, 13, 23, 20, 22],
                     [69, 70, 37, 38, 48, 15, 19, 25, 27, 24],
                     [71, 72, 39, 40, 42, 50, 54, 21, 29, 26],
                     [73, 74, 41, 43, 44, 46, 52, 62, 31, 28],
                     [75, 76, 45, 47, 51, 56, 58, 60, 64, 30],
                     [77, 78, 82, 49, 53, 55, 57, 59, 61, 32],
                     [79, 80, 84, 86, 87, 89, 91, 94, 63, 95],
                     [81, -1, 83, 85, -1, 90, -1, 93, 96, -1]], dtype=int32)
    arrmap = metaloadvariable(session,area,'arrayChannelMap')
    if removebad:
        if (session[:3]=='SPK' and area=='PMv'):
            arrmap = array(arrmap) #remove read-only if present
            arrmap[2,6]=-1
    return arrmap

def get_trial_event(session,area,trial,event):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to use from the session
    event : int
        Which event marker to use
    
    Returns
    -------
    float
        Time-stamp of selected event on selected trial in selected session
    '''
    if dowarn(): 
        print('TRIAL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
        print('EVENT IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    assert trial>0
    debug(trial,event)
    return metaloadvariable(session,area,'eventsByTrial')[trial-1,event-1]

def get_valid_trials(session,area=None):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        A list of trials on which the subject completed the task.
    '''
    if dowarn(): 
        print('TRIAL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    if area is None:
        trials  = set(find(metaloadvariable(session,'M1' ,'eventsByTrial')[:,0])+1)
        trials &= set(find(metaloadvariable(session,'PMv','eventsByTrial')[:,0])+1)
        trials &= set(find(metaloadvariable(session,'PMd','eventsByTrial')[:,0])+1)
        return np.array(list(trials))
    else:
        return np.array(find(1==metaloadvariable(session,area,'eventsByTrial')[:,0]))

def get_available_channels(session,area):
    '''
    Gets the list of channels that are available for a sessiona and area
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    np.array
        availableChannels: which channels in the archive actually exist
        (missing channels may have been electrically bad, or assigned to
        record from a different area)
    '''
    if dowarn(): 
        warn('CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    return np.int64(find(metaloadvariable(session,area,'availableChannels')[:,0])+1)

def get_bad_channels_and_trials(rule='liberal'):
    forbid = {'conservative':'''
    r18m1
    ch 80 79 45
    ch 2
    tr 40
    r18v
    ch 70 18 40 64 83 45 47 96 30 55
    tr 12 38 60 83 93 94
    tr 40 43 47 44
    tr 4 5 6 10 16 17 18 19 20 21 23 24 25 26 27 28 29 30 31 33 34 35 36 37
    r18d
    ch 6 64 27 18 16 14 10 8 22 21 56 92
    tr 4 5 6 10 40 12

    r21m1
    ch 2 4 78 80 79
    tr 51 102 32 93
    r21v
    ch 70 71 66 41 47 42
    tr 83 40 92
    tr 96 94
    r21d
    ch 64 27 22 18 16 14 11 10 8 6 92 90 21
    tr 51 93 102 97 67 26 19 21 56 71 76 77 79 81 82
    tr 2 4 6 8 11 12 13 14 15 16 23 27 25 24 28
    tr 83 40 92
    tr 96 94

    r23m1
    ch 68 77
    tr 86 6 50 85 125
    tr 116 12 14 15
    r23v
    ch 70 55 96 40
    tr 18 60 13 60 29 118 67 123 81
    r23d
    ch 64 61 27 16 14 13 95 21
    tr 6 10 23 85 86 80 100 125 30 33 34 40 43 46 47 48 51  83 87 88 91 92 93 49
    tr 84 77 41 42 37 22 11 32 22
    tr 122 119 115 114 113
    tr 12 14 15 35 44 68 81 116 16

    s18m1
    ch 53 33 49 77 78
    s18v
    ch 40 41 20 44 48 56 62 5
    ch 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26
    tr 142 107

    s24m1
    ch 85 76 75 78
    tr 13
    s24v
    ch 13 40 60 93 25 94 5
    ch 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26
    tr 95 92 6 13
    s24d
    ch 14

    s25m1
    ch 33
    tr 86
    s25v
    ch 60 87 13 40 32 25 14 5
    ch 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26
    tr 86 46
    tr 157 155
    s25d
    ch 14 15 22 56 60
    tr 86
    ''',

    'liberal':'''
    r18m1
    ch 80 79 45 2
    tr 40 83 93 94

    r18v
    ch 70 18 40 64 83 45 47 96 30 55
    tr 40 83 93 94

    r18d
    ch 6 64 27 18 16 14 10 8 22 21 56 92
    tr 40 83 93 94

    r21m1
    ch 2 4 78 80 79

    r21v
    ch 70 71 66 41 47 42

    r21d
    ch 64 27 22 18 16 14 11 10 8 6 92 90 21

    r23m1
    ch 68 77
    tr 86

    r23v
    ch 70 55 96 40
    tr 86

    r23d
    ch 64 61 27 16 14 13 95 21
    tr 86

    s18m1
    ch 53 33 49 77 78

    s18v
    ch 40 41 20 44 48 56 62 5 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26

    s24m1
    ch 85 76 75 78
    tr 6 92

    s24v
    ch 13 40 60 93 25 94 5 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26
    tr 6 92

    s24d
    tr 6 92

    s25m1
    ch 33
    tr 86

    s25v
    ch 60 87 13 40 32 25 14 5 63 89 90 95 96 61 59 58 57 28 32 30 27 83 73 85 87 26
    tr 86

    s25d
    ch 14 15 22 56 60
    tr 86
    '''
    }
    forbid = forbid[rule]

    from collections import defaultdict
    forbidden = defaultdict(list)
    fmonkey  = None
    fsession = None
    farea    = None
    for line in forbid.lower().strip().split('\n'):
        line = line.strip()
        code = line[:2]
        if code=='ch' or code=='tr':
            assert not fsession is None
            assert not farea    is None
            exclude = map(int,line[2:].strip().split())
            forbidden[fsession,farea,code].extend(exclude)
        else:
            if len(line)>=4:
                monkey  = line[0]
                sessnum = line[1:3]
                arcode  = line[3:]
                assert monkey in 'rs'
                fsession = [s for s in flatten(sessionnames) if s.lower()[0]==monkey and s[-2:]==sessnum]
                farea = [a for a in areas if arcode==a.lower() or arcode[0]==a.lower()[-1]]
                assert len(fsession)==1
                assert len(farea)   ==1
                fsession = fsession[0]
                farea    = farea[0]
    for k in forbidden.keys():
        forbidden[k] = sorted(list(set(forbidden[k])))
    return forbidden

def get_bad_channels(session,area,rule='liberal'):
    x = get_bad_channels_and_trials(rule=rule)
    for (s,a,c),v in x.items():
        if (s,a)==(session,area) and c=='ch':
            return v
    return []

def get_bad_trials(session,area=None,rule='liberal'):
    '''
    This will mark a trial as bad if it is marked as bad for any of
    each of the three arrays
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    x = get_bad_channels_and_trials(rule=rule)
    bad = set()
    for (s,a,c),v in x.items():
        if s==session and c=='tr':
            bad|=set(v)
    return sorted(list(set(bad)))


def get_good_channels(session,area,keepBad=False,rule='liberal'):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if dowarn():
        print('CHANNEL IS 1-INDEXED FOR MATLAB COMPATIBILITY')
    good = get_available_channels(session,area)
    if not keepBad:
        for bad in get_bad_channels(session,area,rule=rule):
            good=np.delete(good,find(good==bad))
    return good


def get_good_trials(session,rule='liberal'):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    trials = get_valid_trials(session)
    for a in areas:
        for bad in get_bad_trials(session,a,rule=rule):
            trials=np.delete(trials,find(trials==bad))
    return trials


def get_trial_epoch_in_session_ms(session,area,trial,epoch=None):
    '''
    finds the trial times in ms for given start and stop range relative
    to event.
    epoch = event, start, stop
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    evt = get_trial_event(session,area,trial,e) # event start time relative
    return st+evt,sp+evt


def get_trial_times_ms(session,area,trial,epoch=None):
    '''
    finds the trial times in ms for given start and stop range relative
    to event. These times are not shifted to align with session time,
    they are aligned to object presentation.
    epoch = event, start, stop
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    evt = get_trial_event(session,area,trial,e) # event start time relative
    return arange(st,sp)+evt


def get_session_times_ms(session,area,tr,epoch=None):
    '''
    Trial times are shifted so that the reported times are in session time.
    This means they should agree with spike times relative to the session.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    tm0 = get_trial_event(session,area,tr,4) # trial data start time session
    return get_trial_times_ms(session,area,tr,epoch)+tm0

def get_trial_start_time(session,area,tr):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return get_trial_event(session,area,tr,4)


def get_channel_as_data_index(session,area,ch):
    '''
    Data are always packed in lists of 96 channels even when channels are
    missing. Routines that return "all data" will remove missing channels.
    It is necessary to convert channel index into indecies into the packed
    data
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    available = get_available_channels(session,area)
    # list of channels, 1-indexed, that are available
    # ch will hopefully be among them
    found  = find(available==ch)
    if len(found)!=1:
        print('unusual: ',session,area,ch,found)
    return found[0]


def get_all_pairs_ordered_as_channel_indecies(session,area):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    am = get_array_map(session,area)
    pairs = []
    h,w = shape(am)
    for i in range(h-1):
        for j in range(w-1):
            ch0 = am[i,j]
            ch1 = am[i,j+1]
            ch2 = am[i+1,j]
            if ch0==-1: continue
            p0 = get_channel_as_data_index(session,area,ch0)
            if ch1!=-1:
                p1 = get_channel_as_data_index(session,area,ch1)
                pairs.append(sorted((p0,p1)))
            if ch2!=-1:
                p2 = get_channel_as_data_index(session,area,ch2)
                pairs.append(sorted((p0,p2)))
    pairs = sorted(list(set(map(tuple,pairs))))
    return pairs

def get_data(session,area,trial,event,start,stop,lowf,highf,params):
    """
    Loads neural data from the CGID arrays

    Pseudocode of procedure
    -- identify and locate relevant data archive
    -- import data archive
    -- locate relevant trial, possibly in the raw LFP if we need to use padding
    -- perform requested filtering on all channels
    -- generate time and space bases
    -- extract and pack data

    Note: trial, event, start, stop are using the matlab convention, 1-index

    Note:

    To extract cortical locations of the electrodes, use the same coordinate 
    system as in the videos, where Y is the distance from ventral to dorsal 
    along central sulcus from M1 array in mm. and X is the distance from 
    caudal to rostral from M1 implant in mm.

    Args:
        session (str): session name
        area (str): area name
        trial (int): 1-indexed trial number
        event (int): experimental event number
        start (int): time offset
        stop (int): time offset
        lowf (float): filter start frequency; data is in 1KHz
        highf (float): filter stop frequency; data is in 1KHz
        params (??): filter parameter codes

    Returns:
        times: len nt trial times in ms for the data relative to reference marker
        xys:   nelectrode-x-2 x,y positions of electrodes in array map space
        data:  a nt-x-nelectrode filtered neural data snippit

    Example:
    ::
    
        session = 'RUS120521'
        area    = 'PMv'
        trial   = 10
        event   = 6
        start   = -1000
        stop    = 0
        lowf    = 15
        highf   = 30
        params  = 0
        times,xys,data = get_data(session,area,trial,event,start,stop,lowf,highf,params)
    """
    if dowarn(): print('TRIAL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    print('loading data...'),
    availableChannels = metaloadvariable(session,area,'availableChannels')
    eventsByTrial     = metaloadvariable(session,area,'eventsByTrial')
    print('done')
    trialEvents = eventsByTrial[trial-1]
    trialStart  = trialEvents[3]
    eventCode   = trialEvents[event-1]
    dataStart   = eventCode+trialStart+start
    dataStop    = eventCode+trialStart+stop
    # extracting and filtering the data is the easy part
    times = arange(start,stop)
    data = []
    for ch in get_good_channels(session,area):
        clipped = get_analytic_lfp(session,area,ch,trial,(event,start,stop),lowf,highf,Fs=1000)
        data.append(clipped)
    # compile the list of spatial locations
    box,positions = getElectrodePositions(session,area)
    xys = arr([positions[ch] for ch in sorted(list(positions.keys()))])
    return times,xys,np.array(data).T,squeeze(availableChannels)


def get_data_all_trials(session,area,event,start,stop,lowf,highf,params):
    """
    Loads neural data from the CGID arrays; returns all trials from
    specified experiment. 

    Pseudocode of procedure
    -- identify and locate relevant data archive
    -- import data archive
    -- locate relevant trial, possibly in the raw LFP if we need to use padding
    -- perform requested filtering on all channels
    -- generate time and space bases
    -- extract and pack data

    Note: trial, event, start, stop are using the matlab convention, 1-index

    Note:

    To extract cortical locations of the electrodes, use the same coordinate 
    system as in the videos, where Y is the distance from ventral to dorsal 
    along central sulcus from M1 array in mm. and X is the distance from 
    caudal to rostral from M1 implant in mm.

    Args:
        session (str): session name
        area (str): area name
        trial (int): 1-indexed trial number
        event (int): experimental event number
        start (int): time offset
        stop (int): time offset
        lowf (float): filter start frequency; data is in 1KHz
        highf (float): filter stop frequency; data is in 1KHz
        params (??): filter parameter codes

    Returns:
        times: len nt trial times in ms for the data relative to reference marker
        xys:   nelectrode-x-2 x,y positions of electrodes in array map space
        data:  a nt-x-nelectrode filtered neural data snippit
    """
    global get_data_all_trials_cache
    if (not 'get_data_all_trials_cache' in vars() and not 'get_data_all_trials_cache' in globals()) or get_data_all_trials_cache==None:
        get_data_all_trials_cache = {}
    if (session,area,event,start,stop,lowf,highf,params) in get_data_all_trials_cache:
        return get_data_all_trials_cache[session,area,event,start,stop,lowf,highf,params]
    from scipy.io     import loadmat
    from scipy.signal import butter,filtfilt,lfilter
    archive = '%s_%s.mat'%(session,area)
    print('loading data...')
    eventsByTrial      = metaloadvariable(session,area,'eventsByTrial')
    availableChannels  = metaloadvariable(session,area,'availableChannels')
    UnsegmentedLFP1KHz = metaloadvariable(session,area,'UnsegmentedLFP1KHz')
    print('done')
    NTRIALS,_ = shape(eventsByTrial)
    # design filtering paramters
    # currently 15-30Hz 4th order butterworth zero phase
    # will case on "params" if we ever want to alter this
    assert(params==0)
    FS    = 1000.0
    NYQ   = FS/2.0
    ORDER = 4
    bandwidth  = highf-lowf;
    wavelength = FS/bandwidth;
    padding    = int(ceil(ORDER*wavelength*5/4));
    b,a = butter(ORDER,np.array([lowf,highf])/NYQ,btype='bandpass')
    alldata = []
    for trial in range(1,NTRIALS+1):
        trialEvents = eventsByTrial[trial-1]
        trialStart  = trialEvents[3]
        eventCode   = trialEvents[event-1]
        dataStart   = eventCode+trialStart+start
        dataStop    = eventCode+trialStart+stop
        # extracting and filtering the data is the easy part
        data = []
        for chi,ishere in enumerate(availableChannels):
            if not ishere: continue
            raw = UnsegmentedLFP1KHz[chi][dataStart-padding:dataStop+padding]
            clipped = hilbert(filtfilt(b,a,raw))[padding:-padding]
            data.append(clipped)
        alldata.append(data)
    result = (np.array(alldata).T,availableChannels)
    get_data_all_trials_cache[session,area,event,start,stop,lowf,highf,params] = result
    return result

# load results from disk
def load_ppc_results_archives(directory='.'):
    '''
    DEPRECATED
    
    Results are stored as keys and ppcs. There are 101 PPC values.
    The keys should have 13 variables. See the runppc function for details.
    It may be better to dump the existing PPC archives, update the matlab
    code, and just run everything again.
    
    Example:
    Function header apated from Matlab::
    
            function [ppc,Pxx,F,nTapers] = runppc(
                session,area,unit,obj,grp,event,start,stop,chid,
                perm,permstrategy,jitter,window,bandWidth,subgroup)
    '''
    import os
    results = [f for f in os.listdir(directory) if 'ppc' in f and '.mat' in f and not 'todo' in f]
    allresults = {}
    for f in results:
        session,area = f[:-4].split('_')[1:]
        res  = loadmat(directory+os.sep+f)
        ppcs = res['ppcs']
        keys = res['keys']
        allresults[session,area] = (keys,ppcs)
    return allresults


def get_cgid_unit_info():
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    raise DepricationWarning('Depricated; use allunitsbysession in unitinfo.py instead')
    print('identifying units')
    allfiles=[]
    for parent,dirs,files in os.walk(sortedunits):
        if parent==sortedunits: continue
        if 'low_snr' in parent: continue
        allfiles.extend(files)
    units = []
    for file in allfiles:
        session,area,uid,_,_ = file.split('_')[:5]
        uid = int(uid.split('(')[0])
        if 'SPK' in session:
            nmonkey  = 1
            nsession = 1+spike_sessions.index(session)
        elif 'RUS' in session:
            nmonkey  = 2
            nsession = 1+rusty_sessions.index(session)
        else:
            print('no')
        narea = areas.index(area)+1
        units.append((session,area,uid))
    return units
