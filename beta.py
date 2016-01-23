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

def estimate_beta_band(session,area,bw=8,epoch=None,doplot=False):
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
                    co,fr = cohere(x,y,Fs=1000,NFFT=1024)
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
                        co,fr = cohere(x,y,Fs=1000,NFFT=1024)
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
