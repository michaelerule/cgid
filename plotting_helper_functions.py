#!/usr/bin/python
# -*- coding: UTF-8 -*-
''' 
Helper functions for making figures
'''

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

from   neurotools.graphics.color import parula,extended,isolum
import cgid.spikes
from   neurotools.signal.ppc import pairwise_phase_consistancy
from   scipy.signal.windows import hann

CMAP = parula

def specshow(freqs,specdata,start,stop,FS=1000,aspect='auto',cmap=CMAP):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    assert len(shape(specdata))==2
    NF,NT = shape(specdata)
    try:
        aspect = aspect / FS
    except: pass
    assert NF == len(freqs)
    im = imshow(specdata,aspect=aspect,extent=(0,float(NT)/FS,freqs[0],freqs[-1]),interpolation='bicubic',origin='lower',
        vmin=max(np.min(specdata),mean(specdata)-4*std(specdata)),
        vmax=min(np.max(specdata),mean(specdata)+4*std(specdata)))
    ylabel('Hz')
    xlabel('Time (s)')
    yticks(list(map(round,ylim())))
    set_cmap(cmap)
    xlim((start+1000)/float(FS),(stop+1000)/float(FS))
    ax = gca()
    add_spectrum_colorbar(specdata,cmap)
    sca(ax)

def waveletshow(freqs,specdata,start,stop,FS=1000,aspect='auto',cmap=CMAP):
    '''
    basically specshow -- diffent units for scale bar
    uV^2
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    assert len(shape(specdata))==2
    NF,NT = shape(specdata)
    try:
        aspect = aspect / FS
    except: pass
    assert NF == len(freqs)
    im = imshow(specdata,aspect=aspect,extent=(0,float(NT)/FS,freqs[0],freqs[-1]),interpolation='bicubic',origin='lower',
        vmin=max(np.min(specdata),mean(specdata)-4*std(specdata)),
        vmax=min(np.max(specdata),mean(specdata)+4*std(specdata)))
    ylabel('Hz')
    xlabel('Time (s)')
    yticks(list(map(round,ylim())))
    set_cmap(cmap)
    xlim((start+1000)/float(FS),(stop+1000)/float(FS))
    ax = gca()
    add_wavelet_colorbar(specdata,cmap)
    #gca().yaxis.labelpad = 50
    cbax = gca()
    sca(ax)
    return ax,cbax

def getTrial(session,area,unit,start,stop,trial):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    tsp = ByTrialSpikesMS[unit-1,trial-1]
    tsp = tsp[tsp>=start+1000]
    tsp = tsp[tsp<stop+1000]
    return tsp

def getSTLFP(session,area,unit,start,stop,window=100):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ByTrialLFP1KHz  = metaloadvariable(session,area,'ByTrialLFP1KHz')
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    channelIds      = metaloadvariable(session,area,'channelIds')
    channel         = channelIds[0,unit-1] # channels are also 1 indexed
    channeldata     = ByTrialLFP1KHz[channel-1,0]
    NUNITS,NTRIALS  = shape(ByTrialSpikesMS)
    snippits = []
    for it in range(NTRIALS):
        tsp = ByTrialSpikesMS[unit-1,it]
        tsp = tsp[tsp>=start+1000+window]
        tsp = tsp[tsp<stop+1000-window-1]
        for sp in tsp:
            snippits.append(channeldata[it,sp-window:sp+1+window])
    snippits = array(snippits)
    return snippits

def ppc(session,area,unit,start,stop,
    window=100,FMAX=250,color='k',label=None,nTapers=None,
    lw=1.5,linestyle='-'):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if not nTapers is None:
        warn('WARNING: no longer using multitaper, nTapers will be ignored! using Hann window')
    # depricating original code due to inconsistency with matlab PPC code
    '''
    snippits = getSTLFP(session,area,unit,start,stop,window)
    M        = shape(snippits)[0]
    fs       = fft(snippits)
    raw      = abs(sum(fs/abs(fs),0))**2
    unbiased = (raw-M)/(M**2-M)
    freqs = fftfreq(window*2+1,1./FS)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],color=color,label=label)
    ylim(0,0.5)
    ylabel('PPC',labelpad=-10)
    xlim(0,FMAX)
    nicexy()
    xticks(linspace(0,FMAX,11),['%d'%x for x in linspace(0,FMAX,11)])
    xlabel('Frequency (Hz)')
    title('Pairwise phase consistency')
    simpleaxis(gca())
    '''
    channel = get_unit_channel(session,area,unit)
    '''
    # formerly getting signals from each block. we need to pull the data
    # from the raw LFP though, so that we can grab some LFP outside the
    # blocks in order to analyze spikes close to the edge of block. We
    # need those spikes for statistical power
    signal  = get_good_trial_lfp_data(session,area,channel)
    times   = get_all_good_trial_spike_times(session,area,unit,(6,start,stop))
    '''
    signal = get_raw_lfp_session(session,area,channel)
    times  = get_spikes_session_filtered_by_epoch(session,area,unit,(6,start,stop))

    (freqs,unbiased,phases),snippits = pairwise_phase_consistancy(signal,times,
        window=window,
        Fs=1000,
        multitaper=False,
        biased=False,
        delta=100,
        taper=hann)
    # function does not return anything,
    # just plots (below)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],linestyle,color=color,label=label,lw=lw)
    #cl = ppc_chance_level(nSamples,10000,.999,nTapers)
    #plot(xlim(),[cl,cl],color=color,label=label+' 99.9% chance level')
    #print('chance level is %s'%cl)
    ylim(0,0.5)
    ylabel('PPC',labelpad=-10)
    xlim(0,FMAX)
    nicexy()
    xticks(linspace(0,FMAX,6),['%d'%x for x in linspace(0,FMAX,11)])
    xlabel('Frequency (Hz)')
    title('Pairwise phase consistency')
    simpleaxis(gca())


def compare_ppc_approaches(session,area,unit,start,stop,
    window=100,FMAX=250):
    '''
    Try with
    compare_ppc_approaches('RUS120523','PMv',42,-1000,0,200)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    channel = get_unit_channel(session,area,unit)
    signal = get_raw_lfp_session(session,area,channel)
    times  = get_spikes_session_filtered_by_epoch(session,area,unit,(6,start,stop))

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,
        multitaper=False,biased=False,taper=hann)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Hann unbiased')

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,
        multitaper=False,biased=True,taper=hann)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Hann biased')

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,
        multitaper=True,biased=False,k=1)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Multitaper 1 taper unbiased')

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,k=2,
        multitaper=True,biased=False)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Multitaper 2 taper unbiased')

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,k=3,
        multitaper=True,biased=False)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Multitaper 3 taper unbiased')

    (freqs,unbiased),nSamples = pairwise_phase_consistancy(signal,times,
        window=window,Fs=1000,delta=100,k=4,
        multitaper=True,biased=False)
    use = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],unbiased [use],label='Multitaper 4 taper unbiased')

    #cl = ppc_chance_level(nSamples,10000,.999,nTapers)
    #plot(xlim(),[cl,cl],color=color,label=label+' 99.9% chance level')
    #print('chance level is %s'%cl)
    ylim(0,0.5)
    ylabel('PPC',labelpad=-10)
    xlim(0,FMAX)
    nicexy()
    xticks(linspace(0,FMAX,11),['%d'%x for x in linspace(0,FMAX,11)])
    xlabel('Frequency (Hz)')
    title('Pairwise phase consistency')
    simpleaxis(gca())

def coherence(session,area,unit,window=100,FMAX=250):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    raise NotImplementedError('Coherence function is not implemented; use scipy instead')
    # this is untested, don't use it
    snippits  = getSTLFP(session,area,unit,window)
    M         = shape(snippits)[0]
    fs        = fft(snippits)
    raw2  = (abs(sum(fs,0))/sum(abs(fs),0))**2
    freqs = fftfreq(window*2+1,1./FS)
    use   = (freqs>0.)&(freqs<=250.)
    plot(freqs[use],raw2[use],color='r')
    ylim(0,0.5)
    nicexy()
    ylabel('Coherence',labelpad=-10)
    xlim(0,FMAX)
    xticks(linspace(0,FMAX,11),['%d'%x for x in linspace(0,FMAX,11)])
    xlabel('Frequency (Hz)')
    title('Coherence')

def plotSTA(session,area,unit,start,stop,
    window=100,texttop=False,color='k',label=None,
    lw=1.5,linestyle='-'):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    snippits = getSTLFP(session,area,unit,start,stop,window)
    time = arange(-window,window+1)
    sta  = mean(snippits,0)
    sts  = std(snippits,0)
    N    = shape(snippits)[0]
    sem  = 1.96*sts/sqrt(N)
    plot(time,sta,linestyle,zorder=5,color=color,label=label,lw=lw)
    title('Spike-triggered LFP average')
    xlabel('Time (ms)')
    nicexy()
    simpleraxis(gca())
    xlim(-window,window)
    yextreme = max(*abs(array(ylim())))
    ylim(-yextreme,yextreme)
    yticks([-yextreme,0,yextreme])
    ylabel('STA (µV)')
    gca().yaxis.labelpad = -20
    return snippits

def getRaster(session,area,unit,trial):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    #print('both unit and trial should be 1 indexed')
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    tsp    = ByTrialSpikesMS[unit-1,trial-1]
    y      = zeros(stop-start,dtype=int32)
    y[tsp] = 1
    return y

def plotAllTrials(session,area,unit,start,stop,
    FS=1000.0,s=4,clip_on=False):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    eventsByTrial   = metaloadvariable(session,area,'eventsByTrial')
    cla()
    isgood = find(eventsByTrial[:,0])+1
    NT = stop-start
    NUNITS,NTRIALS = shape(ByTrialSpikesMS)
    isgood = find(eventsByTrial[:,0]) # note 0 indexed deviating from convention
    tally = 0
    xpoints, ypoints = [], []
    for it in range(NTRIALS): # also here
        if not it in isgood: continue
        spikes = getTrial(session,area,unit,start,stop,it+1)
        tally += 1
        xpoints.extend(spikes/FS)
        ypoints.extend(ones(len(spikes))*tally)
    scatter(xpoints,ypoints,
        marker='.',s=s,clip_on=False,color='k',facecolor='k')
    xlim(start/FS+1,stop/FS+1)
    ylabel('Trial No.')
    ylim(0.5,tally+0.5)
    yticks([1,tally])
    noaxis(gca())
    xlabel('Time (s)')

def plotLFP(lfp):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    plot(arange(stop-start)/FS,lfp,'k')
    ylim(-150,150)
    yticks([-150,0,150])
    ylabel('Microvolts')
    xlabel('Time (s)')
    simpleraxis(gca())

def plotLFPSpikes(lfp,start,stop,spikes,
    zoom=1000,vrange=100,FS=1000.0,SPIKECOLOR=(1,0.5,0),lw=2):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    print('caution assumes starting at beginning of trial')
    cla()
    plot((arange(stop-start)/FS)[:zoom],lfp[:zoom],
        'k')#,label='%s-%sHz LFP'%(lowf,highf))
    a,b = ylim()
    ylim(-100,100)
    a,b = ylim()
    spikes = float32(spikes)/FS
    mm = (a+b)/2
    c = (a+mm)/2
    d = (b+mm)/2
    for spt in spikes:
        if spt>=xlim()[1]:continue
        plot([spt,spt],(c,d),color=SPIKECOLOR,lw=lw)
    if len(spikes):
        plot([spt,spt],(c,d),
            color=SPIKECOLOR,lw=lw,label='Spike times')
    yticks([-vrange,0,vrange])
    ylabel('Microvolts')
    xlabel('Time (s)')
    xlim((start+1000)/float(FS),(zoom)/float(FS))
    simpleraxis()
    gca().yaxis.labelpad = -20
    xlabel('Time (s)')
    ylabel('µV')
    # nice_legend()
    legend(frameon=0,borderpad=-1,ncol=2,fontsize=12)

def time_upsample(x,factor=4):
    '''
    Uses fourier transform to upsample x by some factor
    Will detrend, then mirror signal, then FFT,
    then add padding zeros to FFT, then iFFT, then crop, etc, etc
    factor should be an integer (default is 4)
    data is converted to float64
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    assert type(factor) is int
    assert factor>1
    N = len(x)
    dc = mean(x)
    x -= dc
    dx = mean(diff(x))
    x -= arange(N)*dx
    y = zeros(2*N,dtype=float64)
    y[:N]=x
    y[N:]=x[::-1]
    ft = fft(y)
    # note
    # if N is even we have 1 DC and one nyquist coefficient
    # if N is odd we have 1 DC and two nyquist coefficients
    # If there is only one nyquist coefficient, it will be real-values,
    # so it will be it's own complex conjugate, so it's fine if we
    # double it.
    up = zeros(N*2*factor,dtype=float64)
    up[:N//2+1]=ft[:N//2+1]
    up[-N//2:] =ft[-N//2:]
    x = (ifft(up).real)[:N*factor]*factor
    x += dx*arange(N*factor)/float(factor)
    x += dc
    return x

def plotWaveforms(session,area,unit):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    waveForms = get_waveforms(session,area)
    wfs = waveForms[0,unit-1]
    times = arange(48)/30000.*1000*1000 # in uS
    toshow = wfs[:,:100:]
    nshow = shape(toshow)[1]
    for i in range(nshow):
        wf = toshow[:,i]
        wf = time_upsample(wf)
        t  = arange(48*4)/4./30000.*1000000
        dt = t[argmin(wf)]-400
        t -= dt
        plot(t,wf,color=(0.1,0.1,0.1),lw=0.5)
    xlim(times[0],times[-1])
    xlabel(u'μs')
    ylabel(u'μV')
    nicexy()
    noaxis(gca())
    title('Waveforms')
    gca().yaxis.labelpad = -20

def plotISIhist(session,area,unit,start,stop):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    NUNITS,NTRIALS = shape(ByTrialSpikesMS)
    isi = []
    for it in range(NTRIALS):
        isi.extend(diff(getTrial(session,area,unit,start,stop,it+1)))
    hist(isi,linspace(0,120,31),facecolor='k')
    a,b = ylim()
    plot((modefind(isi),)*2,ylim(),color='r',lw=2)
    ylim(a,b)
    nicexy()
    xticks(int32([xlim()[0],int(round(modefind(isi))),xlim()[1]]))
    noaxis(gca())
    xlabel('Time (ms)')
    ylabel('N')
    title('ISI histogram (time)')
    gca().yaxis.labelpad = -20

def isimodefreq(session,area,unit,start,stop,FS=1000):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    NUNITS,NTRIALS = shape(ByTrialSpikesMS)
    isi = []
    for it in range(NTRIALS):
        isi.extend(diff(getTrial(session,area,unit,start,stop,it+1)))
    isi = FS/array(isi)
    isi = isi[~isnan(isi)]
    isi = isi[~isinf(isi)]
    mf = modefind(isi)
    return mf

def plotISIhistHz(session,area,unit,start,stop,
    FS=1000,style='bar',color='k',nbins=30,label=None,
    fmin=0,FMAX=100,
    linestyle='-',lw=2,w=0.8):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ByTrialSpikesMS = metaloadvariable(session,area,'ByTrialSpikesMS')
    NUNITS, NTRIALS = shape(ByTrialSpikesMS)
    isi = []
    for it in range(NTRIALS):
        isi.extend(diff(getTrial(session,area,unit,start,stop,it+1)))
    isi = FS/array(isi)
    if style=='bar':
        hist(isi,linspace(fmin,FMAX,nbins+1),facecolor='k',label=label,rwidth=w)
        a,b = ylim()
        plot((modefind(isi),)*2,ylim(),color='r',lw=lw)
        ylim(a,b)
        nicexy()
        isi = isi[~isnan(isi)]
        isi = isi[~isinf(isi)]
        mf = modefind(isi)
        if (not isnan(mf)) and (not isinf(mf)):
            xticks(int32([xlim()[0],int(round(modefind(isi))),xlim()[1]]))
    elif style=='line':
        h,_ = histogram(isi,bins=nbins,range=(0,100))
        x = linspace(fmin,FMAX,1+nbins)
        x = 0.5*(x[1:]+x[:-1])
        plot(x,h,linestyle,lw=1,color=color,label=label)
    else:
        raise ValueError("Plot style must be line or bar")
    noaxis(gca())
    xlabel('Frequency (Hz)')
    ylabel('N')
    title('ISI histogram\n(frequency)')
    gca().yaxis.labelpad = -20

def getSpec(session,area,unit,trial,epoch,lowf,highf):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    #times,xys,data,availableChannels = get_data(session,area,trial,event,start,stop,lowf,highf,params)
    #channel        = channelIds[0,unit-1] # channels are also 1 indexed
    #chix           = channel2index(channel,availableChannels)
    #channeldata    = data[:,chix]
    #return channeldata,fft_cwt(channeldata,lowf,highf)
    ch = get_channel_id(session,area,unit)
    channeldata = get_raw_lfp(session,area,trial,ch,epoch)
    betadata    = get_filtered_lfp(session,area,trial,ch,epoch,lowf,highf)
    return betadata,\
        neurotools.signal.morlet.fft_cwt(channeldata,lowf,highf,w=10.0)

def plotPPC(unit):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    # PPC is 1 indexed by unit
    fftopt    = 100,10    #(window,bandWidth)
    ch        = 0,
    perminfo  = 0,0,50    #(perm,permstrategy,jitter)
    condition = 3,3       #(obj,grp)
    epoch     = 6,-1000,0 #(event,start,stop)
    basekey   = condition+epoch+ch+perminfo+fftopt
    keys,ppcs = allresults[session,area]
    match1 = find(all(keys==int32((unit,)+basekey+(0,)),1))
    x = squeeze(ppcs[match1])
    #cla()
    plot(ppcfreqs,x,color='k')
    ylim(0,0.5)
    xlim(0,400)
    nicexy()
    xlabel('Frequency (Hz)')
    ylabel('PPC')

def add_wavelet_colorbar(data=None,COLORMAP=extended,
    ax=None,vmin=None,vmax=None):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    # manually add colorbar axes because matplotlib gets it wrong
    if ax is None: ax=gca()
    noaxis()
    bb = ax.get_position()
    x,y,w,h,right,bottom = bb.xmin,bb.ymin,bb.width,bb.height,bb.xmax,bb.ymax
    spacing   = pixels_to_xfigureunits(5)
    cbarwidth = pixels_to_xfigureunits(15)
    cax = axes((right-cbarwidth,bottom-h,cbarwidth,h),axisbg='w',frameon=0)
    sca(cax)
    if vmin is None:
        vmin = np.min(data)
        vmin = np.floor(vmin)
    if vmax is None:
        vmax = np.max(data)
        vmax = np.ceil(vmax)
    print(vmin,vmax)
    imshow(array([linspace(vmax,vmin,100)]).T,
        extent=(vmin,vmax,vmin,vmax),
        aspect='auto',
        cmap=COLORMAP)
    nox()
    noaxis()
    nicey()
    cax.yaxis.tick_right()
    text(
        xlim()[1]+diff(xlim())*2.5,
        mean(ylim()),
        u'µV²',
        fontsize=13,
        rotation=0,
        horizontalalignment='right',
        verticalalignment='center')
    cax.yaxis.labelpad = -11
    cax.yaxis.set_label_position("right")
    sca(ax)
    ax.set_position((x,y,w-spacing-cbarwidth,h))


def add_spectrum_colorbar(data=None,COLORMAP=extended,ax=None,vmin=None,vmax=None):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    # manually add colorbar axes because matplotlib gets it wrong
    if ax is None: ax=gca()
    noaxis()
    bb = ax.get_position()
    x,y,w,h,right,bottom = bb.xmin,bb.ymin,bb.width,bb.height,bb.xmax,bb.ymax
    spacing   = pixels_to_xfigureunits(5)
    cbarwidth = pixels_to_xfigureunits(15)
    cax = axes((right-cbarwidth,bottom-h,cbarwidth,h),axisbg='w',frameon=0)
    sca(cax)
    if vmin is None:
        vmin = np.min(data)
        vmin = np.floor(vmin)
    if vmax is None:
        vmax = np.max(data)
        vmax = np.ceil(vmax)
    print(vmin,vmax)
    imshow(array([linspace(vmax,vmin,100)]).T,
        extent=(vmin,vmax,vmin,vmax),
        aspect='auto',
        cmap=COLORMAP)
    nox()
    noaxis()
    nicey()
    cax.yaxis.tick_right()
    text(
        xlim()[1]+diff(xlim())*3,
        mean(ylim()),
        u'µV²/Hz',
        fontsize=14,
        rotation=0,
        horizontalalignment='right',
        verticalalignment='center')
    cax.yaxis.labelpad = -11
    cax.yaxis.set_label_position("right")
    sca(ax)
    ax.set_position((x,y,w-spacing-cbarwidth,h))

def array_imshow(data,vmin=None,vmax=None,cmap=extended,origin='lower',
    drawlines=1,interpolation='bicubic',extent=(0,4,0,4),ctitle='',W=4,H=4):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if extent!=(0,4,0,4):
        print('different size?')
    if vmin is None:
        vmin=np.nanmin(data)
        print('vmin=',vmin)
    if vmax is None:
        vmax=np.nanmax(data)
        print('vmax=',vmax)
    imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,origin=origin,
        interpolation=interpolation,extent=extent)
    if drawlines:
        for i in linspace(0,10,11):
            axvline(i*4.0/10,color='w',lw=0.3)
            axhline(i*4.0/10,color='w',lw=0.3)
    xlim(0,H)
    ylim(0,W)
    nicex()
    nicey()
    xlabel('mm')
    ylabel('mm')
    fudgex()
    fudgey(3)
    draw()
    cax=good_colorbar(vmin,vmax,cmap,ctitle,sideways=0,spacing=15)
    fudgey(4,cax)
    return cax

def phase_delay_plot(mean_analytic_signal,cm=isolum,UPSAMPLE=50,smooth=2.3,NLINE=6):
    '''
    Accepts an analytic signal map, upsamples it, and plots in the current
    axis the phases. For now, expects a 10x10 array 4x4mm is size.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    W,H = shape(mean_analytic_signal)
    print(W,H)
    Wmm,Hmm = W*0.4, H*0.4
    print(Wmm,Hmm)
    upsampled = dct_upsample(dct_cut_antialias(mean_analytic_signal,smooth),UPSAMPLE)
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = shape(upsampled)
    print(sw, sh)
    fw,fh  = float32(shape(mean_analytic_signal)[:2])*UPSAMPLE
    dw,dh  = (fw-sw)/fw*0.5*Wmm,(fh-sh)/fh*0.5*Hmm
    extent = (dh,dh+sh/fh*Hmm,dw,dw+sw/fw*Wmm)

    # extract angles
    amean  = angle(mean(upsampled))
    angles = (angle(upsampled*exp(-1j*amean))+2*pi+pi+1.75)%(2*pi)
    # plot phase angles
    cax    = array_imshow(angles,0,2*pi,cm,ctitle='Phase (radians)',extent=extent,W=Wmm,H=Hmm)
    # add countours
    for phi in linspace(0,pi,NLINE+1)[:-1]:
        c=contour((angles+phi)%(2*pi),[pi],linewidths=0.8,colors='k',extent=extent)
    # Add title
    title('Average phase delay',fontsize=13)
    # fix up colorbar axis labels
    oldax = gca()
    sca(cax)
    yticks([0,2*pi],['0','$2\pi$'])
    # redraw the plot
    draw()
    sca(oldax)
    xlabel('mm',fontsize=11)
    return cax

def unit_ISI_plot(session,area,unit,epoch=((6,-1000,0),(8,-1000,0)),INFOXPOS=70,LABELSIZE=8,NBINS=20,TMAX=300,INFOYSTART=0,BURST=10):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    spikes = []
    for trial in get_good_trials(session):
        try:
            e,a,b = epoch
            spikes.append(cgid.spikes.get_spikes_event(session,area,unit,trial,e,a,b))
        except:
            for e,a,b in epoch:
                spikes.append(cgid.spikes.get_spikes_event(session,area,unit,trial,e,a,b))
    ISI_events = array(list(flatten(map(diff,spikes))))
    SNR        = cgid.spikes.get_unit_SNR(session,area,unit)
    histc,edges = histogram(ISI_events, bins = linspace(0,TMAX,NBINS+1))
    dx         = diff(edges)[0]
    bar(edges[:-1]+dx*0.1,histc,width=dx*0.8,color=GATHER[-1],edgecolor=(0,)*4)
    allisi = array(ISI_events)
    K   = 20
    x,y = kdepeak(log(K+allisi[allisi>0]))
    x   = exp(x)-K
    y   = y/(K+x)
    y   = y*len(allisi)*dx
    plot(x,y,color=RUST,lw=1.5)
    mean_rate           = sum(array(list(map(len,spikes))))/\
        float(len(get_good_trials(session))*2)
    noburst = allisi[allisi>BURST]
    ISI_cv              = std(noburst)/mean(noburst)
    burstiness          = sum(allisi<BURST)/float(len(allisi))*100
    ll                  = 1./mean(allisi)
    expected_short_isi  = (1.0-exp(-ll*10))*100
    residual_burstiness = burstiness-expected_short_isi
    LH = LABELSIZE+4
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(INFOYSTART   ),'Mean rate = %d Hz'%mean_rate,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(INFOYSTART+LH*1),'ISI CV = %0.2f'%ISI_cv,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(INFOYSTART+LH*2),'SNR = %0.1f'%SNR,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    xlabel('ms',fontsize=LABELSIZE)
    ylabel('No. Events',fontsize=LABELSIZE)
    fudgey(10)
    fudgex(5)
    xlim(0,TMAX)
    nicex()
    nicey()
    simpleaxis()
    title('Monkey %s area %s\nsession %s unit %s'%(session[0],area,session[-2:],unit),loc='center',fontsize=7)


def do_unit_ISI_plot(session, area, unit,
    INFOXPOS = 70, LABELSIZE=8, NBINS=20, FMAX = 250):
    '''
    Add statistical summary printout to the ISI example figures
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    cla()
    spikes = []
    for trial in get_good_trials(session):
        spikes.append(cgid.spikes.get_spikes_event(
            session,area,unit,trial,6,-1000,0))
        spikes.append(cgid.spikes.get_spikes_event(
            session,area,unit,trial,8,-1000,0))
    ISI_events = array(list(flatten(map(diff,spikes))))
    SNR        = get_unit_SNR(session,area,unit)
    histc,edges = histogram(ISI_events, bins = linspace(0,FMAX,NBINS+1))
    dx         = diff(edges)[0]
    bar(edges[:-1]+dx*0.1,histc,
        width=dx*0.8,color=GATHER[-1],edgecolor=(0,)*4)
    allisi = array(ISI_events)
    K   = 20
    x,y = kdepeak(log(K+allisi[allisi>0]))
    x   = exp(x)-K
    y   = y/(K+x)
    y   = y*len(allisi)*dx
    plot(x,y,color=RUST,lw=1.5)
    mean_rate = sum(array(list(map(len,spikes)))/
        float(len(get_good_trials(session))*2))
    ISI_cv              = std(allisi)/mean(allisi)
    burstiness          = sum(allisi<5)/float(len(allisi))*100
    ll                  = 1./mean(allisi)
    expected_short_isi  = (1.0-exp(-ll*10))*100
    mode                = modefind(allisi)
    residual_burstiness = burstiness-expected_short_isi
    LH = LABELSIZE+4
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(20   ),'Mean rate = %d Hz'%mean_rate,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(20+LH*1),'ISI CV = %0.2f'%ISI_cv,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(20+LH*2),'SNR = %0.1f'%SNR,
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    text(INFOXPOS,ylim()[1]-pixels_to_yunits(20+LH*3),
        'Mode = %0.1f ms (%0.1f Hz)'%(mode,1000./mode),
        horizontalalignment='left',
        verticalalignment  ='bottom',fontsize=LABELSIZE)
    axvline(mode,lw=2,color=TURQUOISE)
    xlabel('ms',fontsize=LABELSIZE)
    ylabel('No. Events',fontsize=LABELSIZE)
    fudgey(10)
    fudgex(5)
    xlim(0,FMAX)
    nicex()
    nicey()
    title('Monkey %s area %s\nsession %s unit %s'%(session[0],area,session[-2:],unit),loc='center',fontsize=7)
    return mean_rate,ISI_cv,SNR,mode
