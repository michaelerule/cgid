#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
# more py2/3 compat
from neurotools.system import *
if sys.version_info<(3,):
    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATE

# execute the main figure script first
# this is just to check units

# TODO: repair imports
from neurotools.nlab import memoize

# from cgid.setup import *
# from cgid.plotting_helper_functions import *
# from pylab import *

from cgid.data_loader import metaloadvariable

def unitsum(session,area,unit):
    SUPTITLESIZE = 14
    close('all')
    figure(1,figsize=(10.5,4))
    awf1        = subplot2grid((2,4),(0,0),colspan=1)
    aisiobj1    = subplot2grid((2,4),(0,3),colspan=1)
    aisigo1     = subplot2grid((2,4),(1,3),colspan=1)
    arasterobj1 = subplot2grid((2,4),(0,1),colspan=2)
    arastergo1  = subplot2grid((2,4),(1,1),colspan=2)
    # waveforms and ISI
    sca(awf1)
    plotWaveforms(session,area,unit)
    sca(aisiobj1)
    plotISIhistHz(session,area,unit,-1000,0000)
    sca(aisigo1)
    xlabel('')
    plotISIhistHz(session,area,unit,2000,3000)
    title('')
    sca(arasterobj1)
    plotAllTrials(session,area,unit,-1000,0000)
    xlabel('')
    title('1 second before object presentation', loc='left')
    nicex()
    gca().yaxis.labelpad = -15
    sca(arastergo1)
    plotAllTrials(session,area,unit,2000,3000)
    nicex()
    title('1 second before go cue', loc='left')
    gca().yaxis.labelpad = -15
    subplots_adjust(0.07,0.15,0.93,0.83,.4,.5)
    suptitle('Session %s area %s unit %s'%(session,area,unit),fontsize=SUPTITLESIZE)

def unitsumfig(session,area,unit,savein='./'):
    SUPTITLESIZE = 14
    figure(1,figsize=(10.5,4))
    clf()
    awf1        = subplot2grid((2,4),(0,0),colspan=1)
    aisiobj1    = subplot2grid((2,4),(0,3),colspan=1)
    aisigo1     = subplot2grid((2,4),(1,3),colspan=1)
    arasterobj1 = subplot2grid((2,4),(0,1),colspan=2)
    arastergo1  = subplot2grid((2,4),(1,1),colspan=2)
    # waveforms and ISI
    sca(awf1)
    plotWaveforms(session,area,unit)
    sca(aisiobj1)
    plotISIhistHz(session,area,unit,-1000,0000)
    sca(aisigo1)
    xlabel('')
    plotISIhistHz(session,area,unit,2000,3000)
    title('')
    sca(arasterobj1)
    plotAllTrials(session,area,unit,-1000,0000)
    xlabel('')
    title('1 second before object presentation', loc='left')
    nicex()
    gca().yaxis.labelpad = -15
    sca(arastergo1)
    plotAllTrials(session,area,unit,2000,3000)
    nicex()
    title('1 second before go cue', loc='left')
    gca().yaxis.labelpad = -15
    subplots_adjust(0.07,0.15,0.93,0.83,.4,.5)
    suptitle('Session %s area %s unit %s'%(session,area,unit),fontsize=SUPTITLESIZE)
    savefig(savein+'%s_%s_%s'%(session,area,unit)+'.png')
    draw()

def dumpallunits():
    for s,a in sessions_areas():
        print(s,a)
        NUNITS = len(metaloadvariable(s,a,'unitIds')[0])
        print('No. Units = ',NUNITS)
        for i in range(NUNITS):
            print('\t',s,a,i)
            unitsumfig(s,a,i+1,savein='./unit_info_figures/')
    for (s,a),us in allunitsbysession.iteritems():
        for u in us:
            unitsumfig(s,a,u,savein='./unit_info_figures/')


# further restrictions:
#   No. Spikes pre-obj and pre-go
#   SNR
#   good LFP channel?


def is_unit_on_good_lfp_channel(session,area,unit):
    chs = get_good_channels(session,area)
    chid = get_channel_id(session,area,unit)
    return chid in chs
    
@memoize
def get_unit_snr(session,area,unit):
    wfs = get_waveforms(session,area,unit)
    noise = std(wfs[0])
    signal = 0.5*(mean(np.max(wfs,0)-np.min(wfs,0)))
    return signal/noise
    
def get_total_spikes(session,area,unit,epoch):
    spks = get_spike_times_all_trials(session,area,unit,epoch)
    return sum(map(len,spks))
    
def get_average_rate(session,area,unit,epoch):
    spks = get_spike_times_all_trials(session,area,unit,epoch)
    event,start,stop = epoch
    return mean(map(len,spks))/(stop-start)*1000

def printstats(s,a,u):
    print('\t',s,a,u,end='')
    print('OK ' if is_unit_on_good_lfp_channel(s,a,u) else 'BAD',end='')
    print(get_unit_quality(s,a,u),end='')
    print('SNR=%0.1f'%get_unit_snr(s,a,u),end='')
    print('\t','obj count=%04d rate=%0.1f'%(
        get_total_spikes(s,a,u,preobject),
        get_average_rate(s,a,u,preobject)),end='')
    print('\t','go count=%04d rate=%0.1f'%(
        get_total_spikes(s,a,u,prego),
        get_average_rate(s,a,u,prego)))

def surveyallunits():
    preobject  = 6,-1000,0 # pre-object
    prego      = 8,-1000,0 # pre-go
    allrates  = []
    allcounts = []
    allsnr    = []
    for s,a in sessions_areas():
        print(s,a)
        NUNITS = len(metaloadvariable(s,a,'unitIds')[0])
        print('No. Units = ',NUNITS)
        for u in range(1,1+NUNITS):
            printstats(s,a,u)
            allrates .append(get_average_rate(s,a,u,preobject))
            allrates .append(get_average_rate(s,a,u,prego))
            allcounts.append(get_total_spikes(s,a,u,preobject))
            allcounts.append(get_total_spikes(s,a,u,prego))
            allsnr   .append(get_unit_snr(s,a,u))

def filterunits():
    minsnr   = 5
    mincount = 500
    minrate  = 1
    minqual  = 2
    preobject = 6,-1000,0 # pre-object
    prego     = 8,-1000,0 # pre-go
    for s,a in sessions_areas():
        print(s,a)
        NUNITS = len(metaloadvariable(s,a,'unitIds')[0])
        print('No. Units = ',NUNITS)
        for u in range(1,1+NUNITS):
            if not is_unit_on_good_lfp_channel(s,a,u): continue
            if get_unit_quality(s,a,u)<minqual: continue
            if get_total_spikes(s,a,u,preobject)<mincount: continue
            if get_total_spikes(s,a,u,prego    )<mincount: continue
            if get_average_rate(s,a,u,preobject)<minrate: continue
            if get_average_rate(s,a,u,prego    )<minrate: continue
            if get_unit_snr(s,a,u)<minsnr: continue
            printstats(s,a,u)
            unitsumfig(s,a,u,savein='./unit_info_figures_good/')

def use_best_units():
    global units, allunitsbysession
    usable = os.listdir('./unit_info_figures_good/')
    for s,a in allunitsbysession.keys():
        allunitsbysession[s,a]=[]
        for f in usable:
            _s,_a,_u = f[:-4].split('_')
            if s==_s and a==_a:             
                allunitsbysession[s,a].append(int(_u))
    units = []
    for (session,area), uu in allunitsbysession.iteritems():
        units+=[(session,area,u) for u in uu]

def find_very_good_beta_examples():
    minsnr   = 7
    mincount = 1000
    minrate  = 9
    minqual  = 3
    preobject = 6,-1000,0 # pre-object
    prego     = 8,-1000,0 # pre-go
    for s,a in sessions_areas():
        print(s,a)
        NUNITS = len(metaloadvariable(s,a,'unitIds')[0])
        print('No. Units = ',NUNITS)
        for u in range(1,1+NUNITS):
            if not is_unit_on_good_lfp_channel(s,a,u): continue
            if get_unit_quality(s,a,u)<minqual: continue
            if get_total_spikes(s,a,u,preobject)<mincount: continue
            if get_total_spikes(s,a,u,prego    )<mincount: continue
            if get_average_rate(s,a,u,preobject)<minrate: continue
            if get_average_rate(s,a,u,prego    )<minrate: continue
            if get_unit_snr(s,a,u)<minsnr: continue
            mf1 = isimodefreq(s,a,u,-1000,0000,FS=1000)
            if mf1<15 or mf1>30: continue
            mf2 = isimodefreq(s,a,u, 2000,3000,FS=1000)
            if mf2<15 or mf2>30: continue
            printstats(s,a,u)
            unitsumfig(s,a,u,savein='./unit_info_figures_exemplar/')



    


