import numpy as np

from cgid.lfp                import get_raw_lfp_session, get_filtered_lfp
from cgid.spikes             import get_spikes_session_filtered_by_epoch
from cgid.data_loader        import get_trial_event, good_trials
from neurotools.signal       import bandfilter, get_edges
from neurotools.tools        import wait
from numpy.core.numeric      import convolve
from numpy.lib.function_base import diff
from matplotlib.pyplot       import clf,axvspan,plot,ylim,axhline,draw

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
    threshold   = std(beta_trials)*THSCALE
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

def get_high_low_beta_firing_rates(session,area,unit,epoch,Fs=1000):
    '''
    Computes the unit firing rates during high and low beta events for
    the given task epoch. Good trials only. Defaults to Fs = 1000
    returns threshold, event_rate, nonevent_rate
    '''
    threshold,events = get_high_beta_events(session,area,get_unit_channel(session,area,unit),epoch)
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






