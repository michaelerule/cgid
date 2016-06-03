#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

from neurotools.tools import memoize
from cgid.data_loader import *
from neurotools.signal.signal     import *
from neurotools.signal.multitaper import *
from scipy.signal.signaltools import *
import pickle
import cgid.tools
import cgid.data_loader

#    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),padding)
@memoize
def get_raw_lfp_session(session,area,ch):
    if not ch in cgid.data_loader.get_available_channels(session,area):
        warn('%a is not available channel for %s %s',(ch,session,area))
    if not ch in cgid.data_loader.get_good_channels(session,area):
        warn('%a is not a good channel for %s %s',(ch,session,area))
    return cgid.data_loader.metaloadvariable(
        session,area,'UnsegmentedLFP1KHz')[ch-1,0][:,0]

@memoize
def get_raw_lfp(session,area,trial,channel,epoch,pad=0):
    '''
    get_raw_lfp(session,area,trial,channel,epoch,pad=0)
    '''
    if dowarn():
        print('NOTE CHANNEL IS 1 INDEXED FOR MATLAB COMPATIBILITY')
        print('NOTE TRIAL   IS 1 INDEXED FOR MATLAB COMPATIBILITY')
        print('NOTE EVENT   IS 1 INDEXED FOR MATLAB COMPATIBILITY')
    debug(' get_raw_lfp',trial,channel,epoch)
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
    if onlygood:
        return array([get_raw_lfp(session,area,trial,ch,epoch)\
             for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return array([get_raw_lfp(session,area,trial,ch,epoch)\
             for ch in cgid.data_loader.get_available_channels(session,area)])

def get_good_trial_lfp_data(session,area,channel):
    return array([get_raw_lfp(session,area,channel,trial,(6,-1000,6000))\
         for trial in cgid.data_loader.get_get_good_trials(session)])

def get_all_raw_lfp_all_areas(session,trial,epoch,onlygood=True):
    alllfp = []
    for area in areas:
        alllfp.extend(get_all_raw_lfp(session,area,trial,epoch,onlygood))
    return alllfp

@memoize
def get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000):
    '''
    get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000)
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
    padding    = int(ceil(2.5*wavelength))
    warn('padding %s'%padding)
    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),padding)
    lfp = bandfilter(lfp,fa,fb)
    return lfp[padding:-padding]

@memoize
def get_filtered_lfp_session(session,area,ch,fa,fb,Fs=1000):
    lfp = get_raw_lfp_session(session,area,ch)
    return bandfilter(lfp,fa,fb)

def get_all_filtered_lfp(session,area,tr,epoch,fa,fb,onlygood,Fs=1000):
    if onlygood:
        return array([
            get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs) \
            for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return array([
            get_filtered_lfp(session,area,tr,ch,epoch,fa,fb,Fs) \
            for ch in cgid.data_loader.get_available_channels(
                session,area)])

def get_analytic_lfp(session,area,tr,ch,epoch,fa,fb,Fs=1000):
    if fa is None and fb is None:
        print('need at least one frequency bound')
        assert 0
    if epoch is None: epoch = (6,-1000,6000)
    e,st,sp = epoch
    bandwidth  = fb if fa is None else fa if fb is None else min(fa,fb)
    wavelength = Fs/bandwidth
    padding    = int(ceil(2.5*wavelength))
    warn('padding %s'%padding)
    lfp = get_raw_lfp(session,area,tr,ch,(e,st,sp),pad=padding)
    lfp = bandfilter(lfp,fa,fb)
    return hilbert(lfp)[padding:-padding]

def get_all_analytic_lfp(session,area,tr,epoch,fa,fb,onlygood=False):
    if onlygood:
        return array([get_analytic_lfp(session,area,tr,ch,epoch,fa,fb)\
            for ch in cgid.data_loader.get_good_channels(session,area)])
    else:
        return array([get_analytic_lfp(session,area,tr,ch,epoch,fa,fb)\
            for ch in cgid.data_loader.get_available_channels(session,area)])

@memoize
def get_MUA_lfp(session,area,tr,ch,epoch,fc=250,fsmooth=5):
    debug(ch)
    if ch is None:
        lfp = get_all_analytic_lfp(session,area,tr,epoch,fc,None,True)
        lfp = array([bandfilter(abs(l),fb=fsmooth) for l in lfp])
        lfp = mean(lfp,0)
        return lfp
    else:
        lfp = get_all_analytic_lfp(session,area,tr,ch,epoch,fc,None)
        lfp = bandfilter(abs(lfp),fb=fsmooth)
        return lfp

def get_all_MUA_lfp(session,area,tr,epoch,fc=250,fsmooth=5):
    return array([get_MUA_lfp(session,area,tr,ch,epoch,fc,fsmooth) for ch in cgid.data_loader.get_available_channels(session,area)])

@memoize
def get_mean_lfp(session,area,tr,epoch,fa,fb):
    res = mean(get_all_filtered_lfp(session,area,tr,epoch,fa,fb,True),0)
    return res

@memoize
def get_mean_raw_lfp(session,area,tr,epoch):
    res = mean(get_all_lfp(session,area,tr,epoch),0)
    return res

@memoize
def get_band_envelope(session,area,tr,ch,epoch,fa,fb,fsmooth=5):
    '''
    Amplitude envelope for a specified band.
    '''
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,fa,fb,True)
        if not fsmooth is None: lfp = array([bandfilter(amp(l),fb=fsmooth) for l in lfp])
        else: lfp = array([amp(l) for l in lfp])
        lfp = mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,fa,fb)
        if not fsmooth is None: lfp = bandfilter(amp(lfp),fb=fsmooth)
        else: lfp = array([amp(l) for l in lfp])
        return lfp

def get_beta_power(session,area,tr,ch,epoch,fa,fb,fsmooth=5):
    return get_band_envelope(session,area,tr,ch,epoch,fa,fb,fsmooth)

@memoize
def get_MEP(session,area,tr,ch,epoch,fb):
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
        lfp = mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,None,fb)
        return lfp

@memoize
def get_MEP_average(session,area,tr,epoch,fb=7):
    debug(epoch,tr)
    tr=tr+0
    lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
    lfp = mean(lfp,0)
    return lfp

@memoize
def get_MEP_envelope(session,area,tr,ch,epoch,fb=2,fsmooth=2):
    if ch is None:
        lfp = get_all_filtered_lfp(session,area,tr,epoch,None,fb,True)
        lfp = array([bandfilter(amp(l),fb=fsmooth) for l in lfp])
        lfp = mean(lfp,0)
        return lfp
    else:
        lfp = get_filtered_lfp(session,area,tr,ch,epoch,None,fb)
        lfp = bandfilter(amp(lfp),fb=fsmooth)
        return lfp

@memoize
def get_beta_suppression(session,area,tr,ch,epoch,fa,fb,fc=2,fsmooth=5):
    bpwr = get_beta_power  (session,area,tr,ch,epoch,fa,fb,fsmooth)
    mepp = get_MEP_envelope(session,area,tr,ch,epoch,fc,fsmooth)
    meps = get_MEP         (session,area,tr,ch,epoch,fc)
    muap = get_MUA_lfp     (session,area,tr,ch,epoch,fc=250,fsmooth=fsmooth)
    return zscore(zscore(bpwr)-zscore(mepp)-zscore(muap))

def beta_suppression_summary_plot(session,area,tr,ch):
    '''
    Test example
    plot_beta_suppression = beta_suppression_summary_plot
    ch = cgid.data_loader.get_good_channels(session,area)[0]
    for tr in cgid.data_loader.get_get_good_trials(session,area):
        print(session,area,tr)
        plot_beta_suppression(session,area,tr,ch)
        wait()
        clf()
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
def get_wavelet_transforms(session,area,trial,
    fa=1,fb=55,resolution=1,threads=1):
    lfp = get_all_analytic_lfp(session,area,trial,fa,fb,epoch,onlygood=False)
    freqs,cwt = fft_cwt_transposed(lfp,fa,fb,w=4.0,resolution=resolution,threads=threads)
    return freqs,cwt

@memoize
def get_nearest_neighbor_average_referenced_LFP(session,area,tr,ch,epoch):
    nn = cgid.tools.neighbors(session,area,0)[ch]
    nn_avg = mean([get_lfp(session,area,tr,c,epoch) for c in nn],0)
    lfp = get_lfp(session,area,tr,ch,epoch)
    return lfp-nn_avg


@memoize
def get_nearest_neighbor_average_referenced_LFP_session(session,area,ch):
    nn = cgid.tools.neighbors(session,area,0)[ch]
    lfp = get_raw_lfp_session(session,area,ch)
    nn_avg = mean([get_raw_lfp_session(session,area,c) for c in nn],0)
    return lfp-nn_avg


@memoize
def get_average_band_power_session(session,fa,fb,area):
    '''
    Takes whole session LFP, filtered, gets the Hilbert amplitude per
    channel, returns average over channels.
    function is too slow don't use it.
    '''
    global areas
    use_areas = [area] if not area is None else areas
    channels = []
    for area in areas:
        for ch in cgid.data_loader.good_channels(session,area):
            lfp = get_filtered_lfp_session(session,area,ch,fa,fb)
            channels.append(abs(hilbert(lfp)))
    return mean(channels,0)


@memoize
def signal_history_features(session,area,tr,ch,epoch):
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
    return array(allfeatures)

def get_array_packed_lfp(session,area,trial,epoch):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    '''
    x = get_all_lfp(session,area,trial,epoch,False)
    x = real(cgid.tools.pack_array_data_interpolate(session,area,x))
    return x

def get_array_packed_lfp_filtered(session,area,trial,epoch,fa,fb):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    '''
    x = get_all_filtered_lfp(session,area,trial,epoch,fa,fb,False)
    x = real(cgid.tools.pack_array_data_interpolate(session,area,x))
    return x

@memoize
def get_array_packed_lfp_analytic(session,area,trial,epoch,fa,fb):
    '''
    Retrieves LFP signals and packs them as they are arranged in the
    array. Missing channels are interpolated from nearest neighbors
    '''
    x = get_all_analytic_lfp(session,area,trial,epoch,fa,fb,False)
    x = cgid.tools.pack_array_data_interpolate(session,area,x)
    return x

@memoize
def get_mean_bandfiltered_session(session,epoch,fa,fb):
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
        lfp = [concatenate([cgid.lfp.get_all_analytic_lfp(session, area, tr, epoch, fa, fb, onlygood=True) for area in areas]) for tr in cgid.data_loader.get_get_good_trials(session)]
        lfp = array(lfp)
        mean_beta = mean(lfp,1)
        mean_beta_cache[session,epoch] = mean_beta
        pickle.dump(mean_beta,open('mean_beta_cache.p','wb'))
    return mean_beta

@memoize
def get_MUA_lfp_cached(session,area,tr,ch,epoch,fc,fsmooth):
    debug(ch)
    if ch is None:
        lfp = get_all_analytic_lfp(session,area,tr,epoch,fc,None,True)
        if not fsmooth is None:
            lfp = array([bandfilter(abs(l),fb=fsmooth) for l in lfp])
        else:
            lfp = array([abs(l) for l in lfp])
        lfp = mean(lfp,0)
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
    MUALFP = array([[
        get_MUA_lfp_cached(session,area,tr,None,epoch,250,400)
            for ch in cgid.data_loader.get_good_channels(session,area)]
            for tr in cgid.data_loader.get_get_good_trials(session)])
    return MUALFP

@memoize
def get_MUA_PSD_from_high_frequency_power(session,area,epoch):
    MUALFP    = get_all_good_MUA_unfiltered(session,area,epoch)
    MUA       = MUALFP.reshape((-1,shape(MUALFP)[-1]))
    freqs,specs = multitaper_spectrum(MUA,5)
    return freqs,specs

@memoize
def get_MUA_squared_PSD_from_high_frequency_power(session,area,epoch):
    MUALFP    = get_all_good_MUA_unfiltered(session,area,epoch)
    MUA       = MUALFP.reshape((-1,shape(MUALFP)[-1]))
    freqs,specs = multitaper_squared_spectrum(MUA,5)
    return freqs,specs
