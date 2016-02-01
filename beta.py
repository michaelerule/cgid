'''
Routines related to extracting and processing beta
'''

#def compare_high_low_beta_amplitude_spikes_ppc_5050(session,area,unit,epoch):

from matplotlib.pyplot import *
from numpy import *
from neurotools.plot import *
 
from cgid.data_loader import get_good_trials,get_good_channels
from cgid.lfp import get_filtered_lfp
from cgid.setup import areas

import numpy as np
from cgid.lfp                import get_raw_lfp_session, get_filtered_lfp
from cgid.spikes             import get_spikes_session_filtered_by_epoch,get_unit_channel
from cgid.data_loader        import get_trial_event, good_trials
from neurotools.signal       import bandfilter, get_edges
from neurotools.tools        import wait
from numpy.core.numeric      import convolve
from numpy.lib.function_base import diff
from matplotlib.pyplot       import clf,axvspan,plot,ylim,axhline,draw
from neurotools.tools        import memoize

from scipy.signal.signaltools import *
from numpy import *
from cgid.spikes import *

@memoize
def get_beta_peak(session,area,epoch,fa,fb):
    # determine beta peak
    Fs=1000
    allspec=[]
    for trial in good_trials(session):
        x = get_all_raw_lfp(session, area, trial, epoch)
        f,mts = multitaper_spectrum(x,5,Fs)
        allspec.append(mts)
    allspec     = arr(allspec)
    Ntr,Nch,Nti = shape(allspec)
    meanspec    = mean(allspec,axis=(0,1))
    peaks,vals  = local_maxima(meanspec)
    betapeaks   = peaks[(f[peaks]>fa)&(f[peaks]<fb)]
    betapeak    = f[betapeaks][argmax(meanspec[betapeaks])]
    return betapeak


def estimate_beta_band(session,area,bw=8,epoch=None,doplot=False):
    '''
    return betapeak-0.5*bw,betapeak+0.5*bw
    '''
    print 'THIS IS NOT THE ONE YOU WANT TO USE'
    print 'IT IS EXPERIMENTAL COHERENCE BASED IDENTIFICATION OF BETA'
    assert 0
    if epoch is None: epoch = (6,-1000,3000)
    allco = []
    if not area is None:
        chs = get_good_channels(session,area)[:2]
        for a in chs:
            for b in chs:
                if a==b: continue
                for tr in get_good_trials(session):
                    x = get_filtered_lfp(session,area,a,tr,epoch,None,300)
                    y = get_filtered_lfp(session,area,b,tr,epoch,None,300)
                    co,fr = cohere(x,y,Fs=1000,NFFT=256)
                    allco.append(co)
    else:
        for area in areas:
            chs = get_good_channels(session,area)[:2]
            for a in chs:
                for b in chs:
                    if a==b: continue
                    for tr in get_good_trials(session):
                        x = get_filtered_lfp(session,area,a,tr,epoch,None,300)
                        y = get_filtered_lfp(session,area,b,tr,epoch,None,300)
                        co,fr = cohere(x,y,Fs=1000,NFFT=256)
                        allco.append(co)
    allco = array(allco)
    m = mean(allco,0)
    sem = std(allco,0)/sqrt(shape(allco)[0])
    # temporary in lieu of multitaper
    smooth = ceil(float(bw)/(diff(fr)[0]))
    smoothed = convolve(m,ones(smooth)/smooth,'same')
    use    = (fr<=56)&(fr>=5)
    betafr = (fr<=30-0.5*bw)&(fr>=15+0.5*bw)
    betapeak = fr[betafr][argmax(smoothed[betafr])]
    if doplot:
        clf()
        plot(fr[use],m[use],lw=2,color='k')
        plot(fr[use],smoothed[use],lw=1,color='r')
        plot(fr[use],(m+sem)[use],lw=1,color='k')
        plot(fr[use],(m-sem)[use],lw=1,color='k')
        positivey()
        xlim(*rangeover(fr[use]))
        shade([[betapeak-0.5*bw],[betapeak+0.5*bw]])
        draw()
    return betapeak-0.5*bw,betapeak+0.5*bw


def get_stored_beta_peak(session,area,epoch):
    epochs = [(6, -1000, 0),(8, -1000, 0)]
    if epoch not in epochs:
        print 'supporting onle the 1s pre-obj and pre go'
        print 'epoch',epoch,'not available'
        assert 0
    beta_peaks = {\
     ('RUS120518', 'M1' , (6, -1000, 0)): 16.0,
     ('RUS120518', 'M1' , (8, -1000, 0)): 19.0,
     ('RUS120518', 'PMd', (6, -1000, 0)): 18.0, # changed 26 to 18
     ('RUS120518', 'PMd', (8, -1000, 0)): 18.0,
     ('RUS120518', 'PMv', (6, -1000, 0)): 19.0,
     ('RUS120518', 'PMv', (8, -1000, 0)): 19.0,
     ('RUS120521', 'M1' , (6, -1000, 0)): 16.0,
     ('RUS120521', 'M1' , (8, -1000, 0)): 18.0,
     ('RUS120521', 'PMd', (6, -1000, 0)): 18.0, # changed 24 to 18
     ('RUS120521', 'PMd', (8, -1000, 0)): 18.0,
     ('RUS120521', 'PMv', (6, -1000, 0)): 18.0,
     ('RUS120521', 'PMv', (8, -1000, 0)): 18.0,
     ('RUS120523', 'M1' , (6, -1000, 0)): 16.0,
     ('RUS120523', 'M1' , (8, -1000, 0)): 18.0,
     ('RUS120523', 'PMd', (6, -1000, 0)): 18.0, # changed 23 to 18
     ('RUS120523', 'PMd', (8, -1000, 0)): 18.0,
     ('RUS120523', 'PMv', (6, -1000, 0)): 19.0,
     ('RUS120523', 'PMv', (8, -1000, 0)): 18.0,
     ('SPK120918', 'M1' , (6, -1000, 0)): 20.0,
     ('SPK120918', 'M1' , (8, -1000, 0)): 23.0,
     ('SPK120918', 'PMd', (6, -1000, 0)): 20.0,
     ('SPK120918', 'PMd', (8, -1000, 0)): 23.0,
     ('SPK120918', 'PMv', (6, -1000, 0)): 21.0,
     ('SPK120918', 'PMv', (8, -1000, 0)): 24.0,
     ('SPK120924', 'M1' , (6, -1000, 0)): 21.0,
     ('SPK120924', 'M1' , (8, -1000, 0)): 22.0,
     ('SPK120924', 'PMd', (6, -1000, 0)): 21.0,
     ('SPK120924', 'PMd', (8, -1000, 0)): 23.0,
     ('SPK120924', 'PMv', (6, -1000, 0)): 20.0,
     ('SPK120924', 'PMv', (8, -1000, 0)): 25.0,
     ('SPK120925', 'M1' , (6, -1000, 0)): 20.0,
     ('SPK120925', 'M1' , (8, -1000, 0)): 24.0,
     ('SPK120925', 'PMd', (6, -1000, 0)): 20.0,
     ('SPK120925', 'PMd', (8, -1000, 0)): 24.0,
     ('SPK120925', 'PMv', (6, -1000, 0)): 21.0,
     ('SPK120925', 'PMv', (8, -1000, 0)): 24.0
    }
    return beta_peaks[session,area,epoch]

def get_mean_beta_peak(session,epoch):
    return mean([get_stored_beta_peak(session,a,epoch) for a in areas])
    
def get_mean_beta_peak_full_trial(session):
    return mean([get_stored_beta_peak(session,a,epoch) for a in areas for epoch in [(6,-1000,0),(8,-1000,0)]])
    
    
    
@memoize
def get_high_beta_events(session,area,channel,epoch,
    MINLEN  = 40,   # ms
    BOXLEN  = 50,   # ms
    THSCALE = 1.5,  # sigma (standard deviations)
    lowf    = 10.0, # Hz
    highf   = 45.0, # Hz
    pad     = 200,  # ms
    clip    = True,
    audit   = False
    ):
    '''
    get_high_beta_events(session,area,channel,epoch) will identify periods of 
    elevated beta-frequency power for the given channel.
    
    Thresholds are selected per-channel based on all available trials.
    The entire trial time is used when estimating the average beta power.
    To avoid recomputing, we extract beta events for all trials at once.
    
    By default events that extend past the edge of the specified epoch will
    be clipped. Passing clip=False will discard these events.
    
    returns the event threshold, and a list of event start and stop 
    times relative to session time (not per-trial or epoch time)
    
    passing audit=True will enable previewing each trial and the isolated
    beta events.
    
    >>> thr,events = get_high_beta_events('SPK120925','PMd',50,(6,-1000,0))
    '''

    # get LFP data
    signal = get_raw_lfp_session(session,area,channel)
    
    # esimate threshold for beta events
    beta_trials = [get_filtered_lfp(session,area,channel,t,(6,-1000,0),lowf,highf) for t in good_trials(session)]
    threshold   = np.std(beta_trials)*THSCALE
    print 'threshold=',threshold
    
    N = len(signal)
    event,start,stop = epoch
    all_events = []
    all_high_beta_times = []
    for trial in good_trials(session):
        evt        = get_trial_event(session,area,trial,event)
        trialstart = get_trial_event(session,area,trial,4)
        epochstart = evt + start + trialstart
        epochstop  = evt + stop  + trialstart
        tstart     = max(0,epochstart-pad)
        tstop      = min(N,epochstop +pad)
        filtered   = bandfilter(signal[tstart:tstop],lowf,highf)
        envelope   = abs(hilbert(filtered))
        smoothed   = convolve(envelope,ones(BOXLEN)/float(BOXLEN),'same')
        E = array(get_edges(smoothed>threshold))+tstart
        E = E[:,(diff(E,1,0)[0]>=MINLEN)
                & (E[0,:]<epochstop )
                & (E[1,:]>epochstart)]
        if audit: print E
        if clip:
            E[0,:] = np.maximum(E[0,:],epochstart)
            E[1,:] = np.minimum(E[1,:],epochstop )
        else: 
            E = E[:,(E[1,:]<=epochstop)&(E[0,:]>=epochstart)]
        if audit:
            clf()
            axvspan(epochstart,epochstop,color=(0,0,0,0.25))
            plot(arange(tstart,tstop),filtered,lw=0.7,color='k')
            plot(arange(tstart,tstop),envelope,lw=0.7,color='r')
            plot(arange(tstart,tstop),smoothed,lw=0.7,color='b')
            ylim(-80,80)
            for a,b in E.T:
                axvspan(a,b,color=(1,0,0,0.5))
            axhline(threshold,color='k',lw=1.5)
            xlim(tstart,tstop)
            draw()
            wait()
        all_events.extend(E.T)
        assert all(diff(E,0,1)>=MINLEN)
    return threshold, all_events

@memoize
def get_high_and_low_beta_spikes(session,area,unit,epoch,fa,fb):   
    '''
    threshold, event_spikes, nonevent_spikes = get_high_and_low_beta_spikes(session,area,unit,epoch,ishighbeta)
    '''
    threshold,events = get_high_beta_events(session,area,get_unit_channel(session,area,unit),epoch,lowf=fa,highf=fb)
    spikes = get_spikes_session_filtered_by_epoch(session,area,unit,epoch)
    n_total_spikes     = len(spikes)
    n_total_times      = len(good_trials(session))*(epoch[-1]-epoch[-2])
    events = array(events)
    if shape(events)[0]==0:
        print 'NO EVENTS!!!!!!!'
        return threshold, NaN, Fs*float(n_total_spikes)/n_total_times
    event_spikes = (events[:,1][:,None]>=spikes[None,:])\
                  &(events[:,0][:,None]<=spikes[None,:])
    is_in_event = sum(event_spikes,0)
    return threshold, spikes[is_in_event==1], spikes[is_in_event==0]

@memoize
def get_high_low_ppc_unit(s,a,u,e,fa,fb):
    '''
    High/low beta PPC for a given band.
    The band is used only to identify beta events,
    PPC itself is broad-band.
    freqs, event_ppc, nonevent_ppc, threshold = get_high_low_ppc_unit(s,a,u,e,fa,fb)
    '''
    # get beta LFP.
    # No need to restrict this to high or low beta, 
    # restricting the spikes will accomplish that
    ch = get_channel(s,a,u)
    lfp = get_raw_lfp_session(s,a,ch)
    # get beta events around peak (bottleneck)
    # get spikes separated into high and low beta groups
    threshold, event_spikes, nonevent_spikes = get_high_and_low_beta_spikes(s,a,u,e,fa,fb)
    (freqs,event_ppc   ,_),_ = pairwise_phase_consistancy(lfp,event_spikes   ,window=100,multitaper=False,biased=False,delta=200,taper=hanning)
    (freqs,nonevent_ppc,_),_ = pairwise_phase_consistancy(lfp,nonevent_spikes,window=100,multitaper=False,biased=False,delta=200,taper=hanning)
    return freqs, event_ppc, nonevent_ppc, threshold


@memoize
def get_high_low_beta_firing_rates(session,area,unit,epoch,fa,fb):
    Fs=1000
    '''
    Computes the unit firing rates during high and low beta events for
    the given task epoch. Good trials only. Defaults to Fs = 1000
    returns threshold, event_rate, nonevent_rate
    '''
    threshold,events = get_high_beta_events(session,area,get_unit_channel(session,area,unit),epoch,lowf=fa,highf=fb)
    spikes = get_spikes_session_filtered_by_epoch(session,area,unit,epoch)

    n_total_spikes     = len(spikes)
    n_total_times      = len(good_trials(session))*(epoch[-1]-epoch[-2])

    events = array(events)
    if shape(events)[0]==0:
        print 'NO EVENTS!!!!!!!'
        return threshold, NaN, Fs*float(n_total_spikes)/n_total_times
    # linear time solution exists but quadratic time solution is quicker to code.
    n_event_spikes     = sum((events[:,1][:,None]>=spikes[None,:])
                            &(events[:,0][:,None]<=spikes[None,:]))
    n_nonevent_spikes  = n_total_spikes-n_event_spikes
    n_event_times      = sum(events[:,1]-events[:,0])
    n_nonevent_times   = n_total_times - n_event_times
    event_rate         = Fs*float(n_event_spikes)/n_event_times
    nonevent_rate      = Fs*float(n_nonevent_spikes)/n_nonevent_times
    print session,area,unit,epoch,event_rate,nonevent_rate, threshold
    return threshold, event_rate, nonevent_rate


    
    
    
