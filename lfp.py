#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

from neurotools.tools import memoize
import pickle
import cgid.tools
import cgid.data_loader

import numpy as np
import neurotools.tools

# TODO fix imports
# from cgid.data_loader import *
# from neurotools.signal.signal     import *
# from neurotools.signal.multitaper import *
# from scipy.signal.signaltools import *

#    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),padding)
# from warnings import warn

# Patch for now
def warn(*args,**kwargs):
    print(*args,**kwargs)

from neurotools.signal.signal import bandfilter,hilbert

@memoize
def get_raw_lfp_session(session,area,ch):
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
    if not ch in cgid.data_loader.get_available_channels(session,area):
        warn('%d is not available channel for %s %s'%(ch,session,area))
    if not ch in cgid.data_loader.get_good_channels(session,area):
        warn('%d is not a good channel for %s %s'%(ch,session,area))
    lfp = cgid.data_loader.metaloadvariable(session,area,'UnsegmentedLFP1KHz')[ch-1,0]
    if lfp.shape==(0,0):
        raise ValueError('Requested LFP from nonexistent channel')
    return lfp[:,0]

@memoize
def get_raw_lfp(session,area,trial,channel,epoch,pad=0):
    '''
    get_raw_lfp(session,area,trial,channel,epoch,pad=0)
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    
    '''
    # TODO: fix warning situation
    if neurotools.tools.dowarn():
        print('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
        print('NOTE TRIAL   IS 1 INDEXED FOR MATLAB COMPATIBILITY')
        print('NOTE EVENT   IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    neurotools.tools.debug(' get_raw_lfp',trial,channel,epoch)
    trial = trial+0
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    tm0 = cgid.data_loader.get_trial_event(session,area,trial,4)
    evt = cgid.data_loader.get_trial_event(session,area,trial,e)
    t   = tm0 + evt
    st  = st+t-pad
    sp  = sp+t+pad
    assert st>=0
    assert sp>=0
    lfp = get_raw_lfp_session(session,area,channel)
    return lfp[st:sp]

def get_all_raw_lfp(session,area,trial,epoch,onlygood=True):
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
    if onlygood:
        return np.array([get_raw_lfp(session,area,trial,ch,epoch)\
             for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return np.array([get_raw_lfp(session,area,trial,ch,epoch)\
             for ch in cgid.data_loader.get_available_channels(session,area)])

def get_good_trial_lfp_data(session,area,channel):
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
    return np.array([get_raw_lfp(session,area,trial,channel,(6,-1000,6000))\
         for trial in cgid.data_loader.get_good_trials(session)])

def get_all_raw_lfp_all_areas(session,trial,epoch,onlygood=True):
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
    alllfp = []
    for area in areas:
        alllfp.extend(get_all_raw_lfp(session,area,trial,epoch,onlygood))
    return alllfp

@memoize
def get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000):
    '''
    get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000)
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to use. Trial indexing starts at 1 for Matlab 
        compatibility
    ch : int
        Which channel (electrode) to use. Channel indexing starts at 1 for
        Matlab compatibility
    epoch : int
        Which experiment epoch to use. 
    fa : float
        Low-frequency cutoff in Hz
    fb : float
        High-frequency cutoff in Hz
    Fs : float, default 1000
        Sampling rate in Hz
    
    Returns
    -------
    np.array
        Selected raw LFP, bandpass filtered between frequencies fs and fb
    
    '''
    debug(epoch,tr)
    tr=tr+0
    try: # no time to fix correctly
        e,st,sp = epoch
    except:
        epoch = (6,-1000,6000)
        e,st,sp = epoch
    bandwidth  = fb if fa is None else fa if fb is None else min(fa,fb)
    wavelength = Fs/bandwidth
    padding    = int(np.ceil(2.5*wavelength))
    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),padding)
    lfp = bandfilter(lfp,fa,fb)
    return lfp[padding:-padding]

@memoize
def get_filtered_lfp_session(session,area,ch,fa,fb,Fs=1000):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    ch : int
        Which channel (electrode) to use. Channel indexing starts at 1 for
        Matlab compatibility
    fa : float
        Low-frequency cutoff in Hz
    fb : float
        High-frequency cutoff in Hz
    Fs : float, default 1000
        Sampling rate in Hz
    
    Returns
    -------
    
    '''
    lfp = get_raw_lfp_session(session,area,ch)
    return bandfilter(lfp,fa,fb)

def get_all_filtered_lfp(session,area,tr,epoch,fa,fb,onlygood,Fs=1000):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to use. Trial indexing starts at 1 for Matlab 
        compatibility
    epoch : int
        Which experiment epoch to use. 
    fa : float
        Low-frequency cutoff in Hz
    fb : float
        High-frequency cutoff in Hz
    onlygood : bool
        If true, use only channels specified as `good` by
        `cgid.data_loader.get_good_channels`
    Fs : float, default 1000
        Sampling rate in Hz
    
    Returns
    -------
    
    '''
    if onlygood:
        return np.array([
            get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs) \
            for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return np.array([
            get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs) \
            for ch in cgid.data_loader.get_available_channels(
                session,area)])

def get_analytic_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000):
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
    if fa is None and fb is None:
        print('need at least one frequency bound')
        assert 0
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    bandwidth  = fb if fa is None else fa if fb is None else min(fa,fb)
    wavelength = Fs/bandwidth
    padding    = int(np.ceil(2.5*wavelength))
    warn('padding %s'%padding)
    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),pad=padding)
    lfp = bandfilter(lfp,fa,fb)
    return hilbert(lfp)[padding:-padding]

def get_all_analytic_lfp(session,area,tr,epoch,fa,fb,onlygood=False):
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
    if onlygood:
        return np.array([get_analytic_lfp(session,area,tr,ch,epoch,fa,fb)\
            for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return np.array([get_analytic_lfp(session,area,tr,ch,epoch,fa,fb)\
            for ch in cgid.data_loader.get_available_channels(session,area)])

@memoize
def get_MUA_lfp(session,area,tr,ch,epoch,fc=250,fsmooth=5):
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
    debug(ch)
    if ch is None:
        lfp = get_all_analytic_lfp(session,area,tr,epoch,fc,None,True)
        lfp = np.array([bandfilter(np.abs(l),fb=fsmooth) for l in lfp])
        lfp = np.mean(lfp,0)
        return lfp
    else:
        lfp = get_all_analytic_lfp(session,area,tr,ch,epoch,fc,None)
        lfp = bandfilter(np.abs(lfp),fb=fsmooth)
        return lfp

def get_all_MUA_lfp(session,area,tr,epoch,fc=250,fsmooth=5):
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
    return np.array([get_MUA_lfp(session,area,tr,ch,epoch,fc,fsmooth) for ch in cgid.data_loader.get_available_channels(session,area)])

@memoize
def get_mean_lfp(session,area,tr,epoch,fa,fb):
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
    res = np.mean(get_all_filtered_lfp(session,area,tr,epoch,fa,fb,True),0)
    return res

@memoize
def get_mean_raw_lfp(session,area,tr,epoch):
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
    res = np.mean(get_all_lfp(session,area,tr,epoch),0)
    return res

@memoize
def get_band_envelope(session,area,tr,ch,epoch,fa,fb,fsmooth=5):
    '''
    Amplitude envelope for a specified band.
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    
    '''
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,fa,fb,True)
        if not fsmooth is None: lfp = np.array([bandfilter(amp(l),fb=fsmooth) for l in lfp])
        else: lfp = np.array([amp(l) for l in lfp])
        lfp = np.mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,fa,fb)
        if not fsmooth is None: lfp = bandfilter(amp(lfp),fb=fsmooth)
        else: lfp = np.array([amp(l) for l in lfp])
        return lfp

def get_beta_power(session,area,tr,ch,epoch,fa,fb,fsmooth=5):
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
    return get_band_envelope(session,area,tr,ch,epoch,fa,fb,fsmooth)

@memoize
def get_MEP(session,area,tr,ch,epoch,fb):
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
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
        lfp = np.mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,None,fb)
        return lfp

@memoize
def get_MEP_average(session,area,tr,epoch,fb=7):
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
    debug(epoch,tr)
    tr=tr+0
    lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
    lfp = np.mean(lfp,0)
    return lfp

@memoize
def get_MEP_envelope(session,area,tr,ch,epoch,fb=2,fsmooth=2):
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
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
        lfp = np.array([bandfilter(amp(l),fb=fsmooth) for l in lfp])
        lfp = np.mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,None,fb)
        lfp = bandfilter(amp(lfp),fb=fsmooth)
        return lfp

@memoize
def get_beta_suppression(session,area,tr,ch,epoch,fa,fb,fc=2,fsmooth=5):
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
    bpwr = get_beta_power  (session,area,tr,ch,epoch,fa,fb,fsmooth)
    mepp = get_MEP_envelope(session,area,tr,ch,epoch,fc,fsmooth)
    meps = get_MEP         (session,area,tr,ch,epoch,fc)
    muap = get_MUA_lfp     (session,area,tr,ch,epoch,fc=250,fsmooth=fsmooth)
    return zscore(zscore(bpwr)-zscore(mepp)-zscore(muap))

def beta_suppression_summary_plot(session,area,tr,ch):
    '''
    Example
    -------
    ::
    
        plot_beta_suppression = beta_suppression_summary_plot
        ch = cgid.data_loader.get_good_channels(session,area)[0]
        for tr in cgid.data_loader.get_good_trials(session,area):
            print(session,area,tr)
            plot_beta_suppression(session,area,tr,ch)
            wait()
            clf()
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    
    Returns
    -------
    
    '''
    cla()
    if ch is None:
        plot(get_mean_lfp(session,area,tr),'r')
        plot(get_mean_lfp(session,area,tr,10,45),'b')
    else:
        plot(get_lfp(session,area,tr,ch),'r')
        plot(get_filtered_lfp(session,area,tr,ch,10,45),'b')
    plot(get_MUA_lfp(session,area,tr,ch)*10)
    plot(get_MEP_envelope(session,area,tr,ch))
    plot(get_MEP(session,area,tr,ch))
    plot(get_beta_power(session,area,tr,ch))
    plot(get_beta_suppression(session,area,tr,ch)*10,lw=5)

@memoize
def get_wavelet_transforms(session,area,trial,fa=1,fb=55,resolution=1,threads=1):
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
    lfp = get_all_analytic_lfp(session,area,trial,fa,fb,epoch,onlygood=False)
    freqs,cwt = fft_cwt_transposed(lfp,fa,fb,w=4.0,resolution=resolution,threads=threads)
    return freqs,cwt

@memoize
def get_nearest_neighbor_average_referenced_LFP(session,area,tr,ch,epoch):
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
    nn = cgid.tools.neighbors(session,area,0)[ch]
    nn_avg = np.mean([get_lfp(session,area,tr,c,epoch) for c in nn],0)
    lfp = get_lfp(session,area,tr,ch,epoch)
    return lfp-nn_avg


@memoize
def get_nearest_neighbor_average_referenced_LFP_session(session,area,ch):
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
    nn = cgid.tools.neighbors(session,area,0)[ch]
    lfp = get_raw_lfp_session(session,area,ch)
    nn_avg = np.mean([get_raw_lfp_session(session,area,c) for c in nn],0)
    return lfp-nn_avg


@memoize
def get_average_band_power_session(session,fa,fb,area):
    '''
    Takes whole session LFP, filtered, gets the Hilbert amplitude per
    channel, returns average over channels.
    function is too slow don't use it.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    global areas
    use_areas = [area] if not area is None else areas
    channels = []
    for area in areas:
        for ch in cgid.data_loader.good_channels(session,area):
            lfp = get_filtered_lfp_session(session,area,ch,fa,fb)
            channels.append(abs(hilbert(lfp)))
    return np.mean(channels,0)


@memoize
def signal_history_features(session,area,tr,ch,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to use from the session
    ch : int
        Which channel to examine
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    # note: as currently implemeted this adds a 1ms delay
    # that is, history features are really history features, and
    # never include the current timestep,
    # with the exception of the fact that it's currently set up to use
    scales = [500,250,125,62.5,31.25,15.625,7.8125,3.90625]
    hlen   = 4
    Fs     = 1000
    allfeatures = []
    for nyquist in scales:
        samplerate = 2*nyquist
        sampleskip = Fs/samplerate
        sampletime = sampleskip*hlen
        minfreq    = float(Fs)/sampletime
        event,start,stop = epoch
        if minfreq>2:
            xfilt = get_filtered_lfp(session,area,tr,ch,
            (event,start-sampletime+sampleskip-1,stop),minfreq,nyquist,Fs=1000)
        else:
            xfilt = get_filtered_lfp(session,area,tr,ch,
            (event,start-sampletime+sampleskip-1,stop),None,nyquist,Fs=1000)

        tapped     = [zscore(xfilt[i*sampleskip:-1-(hlen-i-1)*sampleskip]) for i in range(hlen)]
        allfeatures.extend(tapped)
    return np.array(allfeatures)

def get_array_packed_lfp(session,area,trial,epoch):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to use from the session
    epoch : int
        Which event marker to use
    
    Returns
    -------
    
    '''
    x = get_all_lfp(session,area,trial,epoch,False)
    x = real(cgid.tools.pack_array_data_interpolate(session,area,x))
    return x

def get_array_packed_lfp_filtered(session,area,trial,epoch,fa,fb):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : 
        Which trial to use from the session
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    x = get_all_filtered_lfp(session,area,trial,epoch,fa,fb,False)
    x = real(cgid.tools.pack_array_data_interpolate(session,area,x))
    return x

@memoize
def get_array_packed_lfp_analytic(session,area,trial,epoch,fa,fb):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    trial : int
        Which trial to use from the session
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    x = get_all_analytic_lfp(session,area,trial,epoch,fa,fb,False)
    x = cgid.tools.pack_array_data_interpolate(session,area,x)
    return x

@memoize
def get_mean_bandfiltered_session(session,epoch,fa,fb):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    try:
        mean_beta_cache
    except:
        mean_beta_cache = {}
    try:
        mean_beta_cache.update(pickle.load(open('mean_beta_cache.p','rb')))
    except:
        print("no disk cache available")
    if (session,epoch) in mean_beta_cache:
        mean_beta = mean_beta_cache[session,epoch]
    else:
        print('recomputing',session,epoch)
        lfp = [concatenate([cgid.lfp.get_all_analytic_lfp(session, area, tr, epoch, fa, fb, onlygood=True) for area in areas]) for tr in cgid.data_loader.get_good_trials(session)]
        lfp = np.array(lfp)
        mean_beta = np.mean(lfp,1)
        mean_beta_cache[session,epoch] = mean_beta
        pickle.dump(mean_beta,open('mean_beta_cache.p','wb'))
    return mean_beta

@memoize
def get_MUA_lfp_cached(session,area,tr,ch,epoch,fc,fsmooth):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    tr : int
        Which trial to use from the session
    ch : int
        Which channel to examine
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    debug(ch)
    if ch is None:
        lfp = get_all_analytic_lfp(session,area,tr,epoch,fc,None,True)
        if not fsmooth is None:
            lfp = np.array([bandfilter(abs(l),fb=fsmooth) for l in lfp])
        else:
            lfp = np.array([abs(l) for l in lfp])
        lfp = np.mean(lfp,0)
        return lfp
    else:
        lfp = get_analytic_lfp(session,area,tr,ch,epoch,fc,None)
        if not fsmooth is None:
            lfp = bandfilter(abs(lfp),fb=fsmooth)
        else:
            lfp = abs(lfp)
        return lfp

@memoize
def get_all_good_MUA_unfiltered(session,area,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    MUALFP = np.array([[
        get_MUA_lfp_cached(session,area,tr,None,epoch,250,400)
            for ch in cgid.data_loader.get_good_channels(session,area)]
            for tr in cgid.data_loader.get_good_trials(session)])
    return MUALFP

@memoize
def get_MUA_PSD_from_high_frequency_power(session,area,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    MUALFP    = get_all_good_MUA_unfiltered(session,area,epoch)
    MUA       = MUALFP.reshape((-1,shape(MUALFP)[-1]))
    freqs,specs = multitaper_spectrum(MUA,5)
    return freqs,specs

@memoize
def get_MUA_squared_PSD_from_high_frequency_power(session,area,epoch):
    '''
    
    Parameters
    ----------
    session : string
        Which experimental session to use, for example "SPK120924"
    area : string
        Which motor area to use, for example 'PMv'
    epoch : 
        Which event marker to use
    
    Returns
    -------
    
    '''
    MUALFP    = get_all_good_MUA_unfiltered(session,area,epoch)
    MUA       = MUALFP.reshape((-1,shape(MUALFP)[-1]))
    freqs,specs = multitaper_squared_spectrum(MUA,5)
    return freqs,specs
