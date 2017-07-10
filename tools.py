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

from collections import defaultdict
from matplotlib.cbook   import flatten
from neurotools.tools   import memoize
from neurotools.spatial.array import trim_array, pack_array_data

import cgid.lfp
import cgid.array

# TODO fix imports
#from scipy.stats.stats  import *
#from matplotlib.pyplot  import *
#from neurotools.getfftw import *
#from neurotools.plot    import *
#from cgid.config        import *
#from cgid.data_loader   import *

def overlay_markers(c1='w',c2='k',FS=1000.,nevents=3,fontsize=14,npad=None,labels=None,clip_on=False):
    '''
    Args:
        labels: default none. Can be
            "markers" for symbols or
            "names" for ['Object presented','Grip cued','Go cue']
            "short" for ['Object','Grip','Go cue']
    '''
    a,b = xlim()
    dx,dy = get_ax_pixel()
    debug( 'dx,dy=%s,%s'%(dx,dy))
    if labels=='markers':
        names = ['$\circ$','$\diamond$','$\star$']
    elif labels=='names':
        names = ['Object presented','Grip cued','Go cue']
    elif labels=='short':
        names = ['Object','Grip','Go cue']
    else:
        names = (None,)*3
    locations = [1000,2000,4000]
    for time,label in zip(locations,names)[:nevents]:
        time = float(time)/FS
        if time<=a: continue
        if time>=b: continue
        plot([time,time],ylim(),color=c2,lw=3,zorder=Inf,clip_on=clip_on)
        plot([time,time],ylim(),color=c1,lw=1,zorder=Inf,clip_on=clip_on)
        if not labels is None:
            text(time,ylim()[1]+dy*4,label,
                rotation=0,color='k',fontsize=fontsize,
                horizontalalignment='center',verticalalignment='bottom')
    xlim(a,b)

overlay_events = overlay_markers

def channel2index(channel,availableChannels):
    if dowarn(): print('NOTE EXPECTING 1 INDEXED CHANNEL')
    if dowarn(): print('RETURNING 0 INDEX OF DATA ARRAY')
    availableChannels = squeeze(availableChannels)
    return cumsum(availableChannels)[channel-1]-1

def unit_channel_as_index(unit):
    return channel2index(channelIds[0,unit-1],availableChannels)

def get_all_isi_epoch(session,area,unit,epoch):
    '''
    Returns the durations of all ISI intervals from the given session, area,
    unit, and epoch
    '''
    all_isi = []
    for trial in cgid.data_loader.get_valid_trials(session,area):
        spk = get_spikes_epoch(session,area,unit,trial,epoch)
        isi = diff(spk)
        all_isi.extend(isi)
    return array(all_isi)

def get_burstiness(session,area,unit,epoch,thr=5):
    isi = get_all_isi_epoch(session,area,unit,'obj')
    return sum(isi<thr)/float(len(isi))

@memoize
def get_all_ibi_epoch(session,area,unit,epoch,thr):
    if dowarn(): print('note: threshold is in ms, defaults to 5')
    all_ibi = []
    if thr==None: thr=5
    for trial in cgid.data_loader.get_valid_trials(session,area):
        spk   = get_spikes_epoch(session,area,unit,trial,epoch)
        if spk==None:
            warn('ERROR %s %s %s %s'%(session,area,unit,epoch))
            assert 0
        if len(spk)<2: continue
        isi   = diff(spk)
        burst = spk[find(isi<=thr)]
        if len(burst)>2:
            all_ibi.extend(diff(burst))
    return array(all_ibi)

@memoize
def get_all_ibi_merged_epoch(session,area,unit,epoch,thr):
    if dowarn(): print('note: threshold is in ms, defaults to 5')
    if dowarn(): print('note: misses burst if starts on 1st spike, TODOFIX')
    all_ibi = []
    if thr==None: thr=5
    for trial in cgid.data_loader.get_valid_trials(session,area):
        spk   = get_spikes_epoch(session,area,unit,trial,epoch)
        if spk==None:
            warn('ERROR %s %s %s %s'%(session,area,unit,epoch))
            assert 0
        if len(spk)<3: continue
        isi   = diff(spk)
        short = isi<thr
        burst = spk[find(diff(int32(short))==1)+1]
        if len(burst)>2:
            all_ibi.extend(diff(burst))
    return array(all_ibi)

def sessions_areas():
    for s in flatten(sessionnames):
        for a in areas:
            yield s,a

def quicksessions():
    for s in [x[0] for x in sessionnames]:
        for a in areas:
            yield s,a

def fftppc(snippits):
    M,window = shape(snippits)
    if M<=window and dowarn(): print('WARNING SAMPLES SEEM TRANSPOSED?')
    assert M>window
    fs = ftw(snippits)
    raw      = abs(sum(fs/abs(fs),0))**2
    unbiased = (raw/M-1)/(M-1)
    freqs    = fftfreq(window,1./FS)
    return freqs[:(window+1)/2], unbiased[:(window+1)/2]

def ch2chi(session,area,ch):
    '''
    Some electrode banks are split over two arrays.
    '''
    assert ch>0
    all_available = cgid.data_loader.get_available_channels(session,area)
    chi = find(all_available==ch) # note zero indexing!
    if not len(chi)==1:
        warn('unusual',(session,area,ch,all_available,chi))
        assert 0
    return chi[0]

def pack_array_data_interpolate(session,area,data):
    '''
    Accepts a collection of signals from array channels, as well as
    an array map containing indecies (1-indexed for backwards compatibility
    with matlab) into that list of channel data.

    This will interpolate missing channels as an average of nearest
    neighbors.

    :param data: should be a NChannel x Ntimes array
    :param session: the session corresponding to data, needed for array map
    :param area: the area corresponding to the data, needed to get array map
    :return: returns LxKxNtimes 3D array of the interpolated channel data
    '''
    arrayMap = cgid.data_loader.get_array_map(session,area)
    def pack_array_data(data,arrayMap):
        '''
        Accepts a collection of signals from array channels, as well as
        an array map containing indecies (1-indexed for backwards compatibility
        with matlab) into that list of channel data.

        This will interpolate missing channels as an average of nearest
        neighbors.

        :param data: NChannel x Ntimes array
        :param arrayMap: array map, 1-indexed, 0 for missing electrodes
        :return: returns LxKxNtimes 3D array of the interpolated channel data
        '''
        # first, trim off any empty rows or columns from the arrayMap
        arrayMap = trim_array(arrayMap)
        # prepare array into which to pack data
        L,K   = shape(arrayMap)
        NCH,N = shape(data)
        packed = zeros((L,K,N),dtype=complex64)
        J = shape(data)[0]
        M = sum(arrayMap>0)
        if J!=M:
            warn('bad: data dimension differs from number of array electrodes')
            warn('data %s array'%((J,M),))
            warn('this may just be because some array channels are removed')
        for i,row in enumerate(arrayMap):
            for j,ch in enumerate(row):
                # need to convert channel to channel index
                if ch==-1:
                    # we will need to interpolate from nearest neighbors
                    ip = []
                    if i>0  : ip.append(arrayMap[i-1,j])
                    if j>0  : ip.append(arrayMap[i,j-1])
                    if i+1<L: ip.append(arrayMap[i+1,j])
                    if j+1<K: ip.append(arrayMap[i,j+1])
                    ip = [ch for ch in ip if ch>0]
                    assert len(ip)>0
                    for chii in ip:
                        qi = ch2chi(session,area,chii)
                        packed[i,j,:] += data[qi,:]
                    packed[i,j,:] *= 1.0/len(ip)
                else:
                    assert ch>0
                    packed[i,j,:] = data[ch2chi(session,area,ch),:]
        return packed
    return pack_array_data(data,arrayMap)

@memoize
def onarraydata(function,session,area,trial,epoch,fa,fb):
    '''
    Applies functions that accept AxBxN data where AxB are array indecies.
    and N is timepoints over the specified section of data.

    Note: array interp routines take array map from get_array_map.
    Bad channels can be removed by removing them in this function.
    '''
    frames = cgid.array.get_analytic_frames(session,area,trial,epoch,fa,fb)
    A,B,N  = shape(frames)
    result = function(frames)
    if len(shape(result))==2:
        result = arr(result)
        a,b = shape(result)
        if a<b: result=result.T
    M     = shape(result)[0]
    times = cgid.data_loader.get_trial_times_ms(session,area,trial,epoch)
    T     = len(times)
    assert T==N
    if M>N:
        debug(map(str,(A,B,N,M,T)))
        assert M<=N
    if M<N:
        warn('warning input len %s output len %s assuming cropped'%(N,M))
        difference = N-M
        startcrop = difference//2
        stopcrop = -(difference-startcrop)
        times = times[startcrop:stopcrop]
    return times,result

def ondatadata(function,session,area,trial,epoch,fa,fb):
    '''
    '''
    # just a temp hack no time no time for better
    debug(epoch,trial)
    trial=trial+0
    debug( '>>> calling %s %s %s %s'%(function.__name__,session,area,trial))
    result = function(session,area,trial,epoch,fa,fb)
    if len(shape(result))==2:
        result = arr(result)
        a,b = shape(result)
        if a<b: result=result.T
    M      = shape(result)[0]
    times  = cgid.data_loader.get_trial_times_ms(session,area,trial,epoch)
    N      = len(times)
    if M>N:
        print(M,N,shape(times),shape(result))
    assert M<=N
    if M<N:
        warn('warning input len %s output len %s assuming cropped'%(N,M))
        difference = N-M
        startcrop = difference//2
        stopcrop = -(difference-startcrop)
        times = times[startcrop:stopcrop]
    return times,result


def ofrawdata(function,session,area,trial,epoch):
    # just a temp hack no time no time for better
    debug(epoch,trial)
    trial=trial+0
    debug( '>>> calling %s %s %s %s'%(function.__name__,session,area,trial))
    result = function(session,area,trial,epoch)
    if len(shape(result))==2:
        result = arr(result)
        a,b = shape(result)
        if a<b: result=result.T
    M      = shape(result)[0]
    times  = cgid.data_loader.get_trial_times_ms(session,area,trial,epoch)
    N      = len(times)
    if M>N:
        print(M,N,shape(times),shape(result))
    assert M<=N
    if M<N:
        warn('warning input len %s output len %s assuming cropped'%(N,M))
        difference = N-M
        startcrop = difference//2
        stopcrop = -(difference-startcrop)
        times = times[startcrop:stopcrop]
    return times,result

def onpopdata(statistic,session,area,trial,epoch,fa,fb):
    '''
    Extracts data and applies provided function 'statistic' to it.
    Returns time base and statistic over time.
    Times may vary as some statistics clip or truncate the data.

    Statistic must accept one argument "data" which is a K channel by N time
    complex matrix of analytic signals.

    Statistic must return a length M<=N vector. If M<N, this function assumes
    that the data were truncated symmetrically to generate a cropped time
    base.

    If not provided, session, area, and trial will be taken from 
    globals ( if they are not present in globals it will crash )
    
    If not provided, epoch is 6,-1000,6000 ( the whole trial )

    TODO: support functions that returnmultiple values

    Example:
    >>> x = onpopdata(sliding_population_signal_coherence)
    '''
    if epoch is None:
        epoch = 6,-1000,6000
    debug( '>>> epoch is %s'%(epoch,))
    lfp = cgid.lfp.get_all_analytic_lfp(session,area,trial,epoch,fa,fb)
    K,N = shape(lfp)
    processed = statistic(lfp)
    assert len(shape(processed))==1
    M = shape(processed)[0]
    times = cgid.data_loader.et_trial_times_ms(session,area,trial,epoch)
    T = len(times)
    assert T==N
    assert M<=N
    if M<N:
        warn('warning input len %s output len %s assuming cropped'%(N,M))
        difference = N-M
        startcrop = difference//2
        stopcrop = -(difference-startcrop)
        times = times[startcrop:stopcrop]
    return times,processed

def overdata(statistic,session,area,trial,epoch,fa,fb):
    '''
    This applies the routine "statistic" over data. Statistic can either
    take a K x N array of population analytic signal data, or an A x B x N
    of array-positioned data. This function inspects the provided
    "statistic" function. if it starts with "array\_" it applies the function
    to the array packed data. if it starts with "population\_" it applies
    the function to population vector data. otherwise, an error is thrown.

    note: static typing and pattern matching to array dimension signatures
    could render this necessray. sadly a language feature python lacks,
    afaik.
    '''
    trial += 0
    fname = statistic.__name__
    tag = fname.split('_')[0]
    debug( '>>> tag is %s'%tag)
    debug( '>>> epoch is %s'%(epoch,))
    if tag=='array':
        return onarraydata(statistic,session,area,trial,epoch,fa,fb)
    elif tag=='population':
        return onpopdata(statistic,session,area,trial,epoch,fa,fb)
    elif tag=='data':
        return ondatadata(statistic,session,area,trial,epoch,fa,fb)
    elif tag=='get':
        return ofrawdata(statistic,session,area,trial,epoch)
    assert 0

@memoize
def onsession(statistic,session,area,epoch,fa,fb):
    '''
    applies statistic over all trials in the session.
    doesn't return exact times since events have different timings on each
    trial. instead returns an event-relative time base.
    times,res=onsession(array_average_ampltiude)
    '''
    try:
        e,st,sp  = epoch
    except:
        epoch = 6,-1000,6000
        e,st,sp  = epoch
    debug('!!>>> applying'+' '.join(map(str,(statistic,session,area,epoch,fa,fb))))
    trials  = cgid.data_loader.get_good_trials(session)
    print(trials)
    results = [overdata(statistic,session,area,tr,epoch,fa,fb) for tr in trials]
    results = arr(results)
    rtimes  = results[0,0,:]
    data    = results[:,1,:]
    times   = arange(st,sp)
    N = len(times)
    M = len(rtimes)
    if N>M:
        warn('warning input len %s output len %s assuming cropped'%(N,M))
        difference = N-M
        startcrop = difference//2
        stopcrop = -(difference-startcrop)
        times = times[startcrop:stopcrop]
    return times,data


def onsession_summary_plot(statistic,session,area,fa,fb,color1=None,color2=None,colors=None,drawCues=False,smoothat=None,dolegend=False):
    '''
    onsession_summary_plot(array_average_ampltiude)
    '''
    epoch = (6,-1000,6000)
    debug('!>>> applying',statistic,session,area,epoch,fa,fb)
    times,res=onsession(statistic,session,area,epoch,fa,fb)
    cla()

    filter_function = box_filter
    if not smoothat is None:
        res = arr([filter_function(x,smoothat) for x in res])

    offset = (7000-len(times))/2
    times = arange(len(times))+offset
    ss = sort(res,axis=0)
    N = shape(res)[0]
    p0 = int(N*10/100+.4)
    p9 = int(N*90/100+.4)
    q1 = N/4
    q3 = N*3/4
    Q1 = ss[q1,:]
    Q2 = ss[q3,:]
    P0 = ss[p0,:]
    P9 = ss[p9,:]
    m = mean(res,0)
    M = median(res,0)
    s = std(res,0)
    sem = 1.96*s/sqrt(N)

    if not smoothat is None:
        m   = filter_function(m,smoothat)
        M   = filter_function(M,smoothat)
        s   = filter_function(s,smoothat)
        sem = filter_function(sem,smoothat)
        Q1 = filter_function(Q1,smoothat)
        Q2 = filter_function(Q2,smoothat)
        P0 = filter_function(P0,smoothat)
        P9 = filter_function(P9,smoothat)

    plot(times,M,color=color1,lw=5,zorder=0,label='Interquartile range')
    fill_between(times,Q1,Q2,color=color1,lw=0,zorder=2)
    plot(times,M,color='k',lw=2,zorder=2,label='Median') #color2

    if dolegend: nicelegend()
    name = ' '.join(statistic.__name__.replace('_',' ').split()).title()
    title('%s, %s-%sHz, %s %s'%(name,fa,fb,session,area))
    xlabel('Time (ms)')
    ylabel(name)
    xlim(times[0],times[-1])
    if drawCues:
        overlayEvents('k','w',FS=1)
    draw()



def allsession_summary_plot(statistic,monkey,area,fa,fb,color1=None,color2=None,colors=None,drawCues=False,smoothat=None,dolegend=False):
    '''
    allsession_summary_plot(array_average_ampltiude)
    '''
    filter_function = box_filter
    epoch = (6,-1000,6000)
    debug('!>>> applying',statistic,monkey,area,epoch,fa,fb)
    all_res = []
    for _s,_a in sessions_areas():
        if _s[0]!=monkey[0]: continue
        if _a!= area: continue
        times,res=onsession(statistic,_s,area,epoch,fa,fb)
        if not smoothat is None:
            res = arr([filter_function(x,smoothat) for x in res])
        all_res.extend(res)

    offset = (7000-len(times))/2
    times = arange(len(times))+offset
    ss = sort(res,axis=0)
    N = shape(res)[0]
    p0 = int(N*10/100+.4)
    p9 = int(N*90/100+.4)
    q1 = N/4
    q3 = N*3/4
    Q1 = ss[q1,:]
    Q2 = ss[q3,:]
    P0 = ss[p0,:]
    P9 = ss[p9,:]
    m = mean(res,0)
    M = median(res,0)
    s = std(res,0)
    sem = 1.96*s/sqrt(N)

    if not smoothat is None:
        m   = filter_function(m,smoothat)
        M   = filter_function(M,smoothat)
        s   = filter_function(s,smoothat)
        sem = filter_function(sem,smoothat)
        Q1 = filter_function(Q1,smoothat)
        Q2 = filter_function(Q2,smoothat)
        P0 = filter_function(P0,smoothat)
        P9 = filter_function(P9,smoothat)

    #plot(times,M,color=color1,lw=5,zorder=0,label='Interquartile range')
    fill_between(times,Q1,Q2,color=color1,lw=0,zorder=2)
    plot(times,M,color=color2,lw=2,zorder=2,label='Median')

    if dolegend: nicelegend()
    name = ' '.join(statistic.__name__.replace('_',' ').split()).title()
    title('%s, %s-%sHz, %s %s'%(name,fa,fb,monkey,area))
    xlabel('Time (ms)')
    ylabel(name)
    xlim(times[0],times[-1])
    if drawCues:
        overlayEvents('k','w',FS=1)
    draw()

    return m,s,sem,M,Q1,Q2,P0,P9


def ontrial_summary_plot(statistic,session,area,trial,epoch,fa,fb):
    '''
    ontrial_summary_plot(array_average_ampltiude)
    '''
    name = ' '.join(statistic.__name__.replace('_',' ').split()).title()
    assert epoch is None
    times,res=overdata(statistic,session,area,trial,epoch,fa,fb)
    cla()
    N = randint(100)
    c1 = lighthues(100)[N]
    c2 = darkhues(100)[N]
    plot(times,res,lw=2,color=c2)
    title('%s, %s-%sHz, %s %s'%(name,fa,fb,session,area))
    xlabel('Time (ms)')
    ylabel(name)
    xlim(times[0],times[-1])
    overlayEvents('k','w',FS=1)
    #tight_layout()
    draw()

def neighbors(session,area,onlygood):
    '''
    Constructs an adjacency graph for a given array. Each channel
    number is mapped to a list of the up to 4 channels immediately
    adjacent to it. If argument "onlygood" is true, then bad channels
    are excluded from this map.
    '''
    am = cgid.data_loader.get_array_map(session,area,removebad=False)
    if onlygood:
        goodch   = cgid.data_loader.get_good_channels(session,area)
        maskedam = -ones(shape(am),dtype=int32)
        for c in goodch: maskedam[am==c]=c
        am = maskedam
    else:
        goodch = cgid.data_loader.get_available_channels(session,area)
    N,M = shape(am)
    neighbors = defaultdict(list)
    for c in goodch:
        i,j = squeeze(where(am==c))
        if i-1>=0 and am[i-1][j]!=-1: neighbors[c].append(am[i-1][j])
        if i+1<N  and am[i+1][j]!=-1: neighbors[c].append(am[i+1][j])
        if j-1>=0 and am[i][j-1]!=-1: neighbors[c].append(am[i][j-1])
        if j+1<M  and am[i][j+1]!=-1: neighbors[c].append(am[i][j+1])
    return neighbors
