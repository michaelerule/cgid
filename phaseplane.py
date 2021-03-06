#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Phase plane plotting routines for LFP phase analysis
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
from neurotools.system import *
if sys.version_info<(3,):
    from itertools import imap as map

from cgid.data_loader  import *
from cgid.lfp          import *
from neurotools.tools  import memoize,globalize,warn,tic,toc
from matplotlib.pyplot import *
from neurotools.graphics.plot   import *
from neurotools.graphics.color  import *
from neurotools.stats.circular import *

import matplotlib as plt
import numpy as np

def phase_plane_animation(session,tr,areas,fa=10,fb=45,epoch=None,\
    skip=1,saveas=None,hook=None,FPS=30,stabilize=True,extension='.pdf',markersize=1.5,figtitle=None):
    '''
    phase_plane_animation(session,tr)
    
    Parameters
    ----------
    
    '''
    warn('Also plots "bad" channels')
    areacolors = [OCHRE,AZURE,RUST]
    print(session, tr)
    # save current figure so we can return to it
    ff=gcf()
    ax=gca()
    # get time base
    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]
    # retrieve filtered array data
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    # compute phase velocities for stabilization
    alldata = concatenate(data.values(),1)
    phasedt = rewrap(diff(angle(alldata),1,0))
    phasev  = median(phasedt,axis=1)
    phaseshift = append(0,cumsum(phasev))
    # compute stabilization differently, using median phase
    phaseshift = angle(mean(alldata,axis=1))
    # PREPARE FIGURE
    if figtitle==None:
        figtitle = 'Analytic Signal %s-%sHz\n%s trial %s'%(fa,fb,session,tr)
    if not saveas is None:
        saveas += '_%s_%s_%s_%s'%(session,tr,fa,fb)
    figure('Phase Plane')
    a2 = cla()
    M = 10
    for a in areas:
        M = max(M,int(ceil(np.max(abs(data[a]))/10.))*10)
    complex_axis(M)
    title(figtitle+' t=%dms'%times[0])
    # prepare output directory if we're going to save this
    if not saveas is None:
        savedir = './'+saveas
        ensuredir(savedir)
    # initialize points
    scat={}
    for i,a in en(areas):
        scat[a] = scatter(*c2p(data[a][0]),s=markersize**2,color=areacolors[i],label=a)
    nice_legend()
    # perform animation
    st = now()
    for i,t in en|times:
        stabilizer = exp(-1j*phaseshift[i]) if stabilize else 1
        for a in areas:
            scat[a].set_offsets(c2p(stabilizer*data[a][i]).T)
        title(figtitle+' t=%sms'%t)
        draw()
        if not saveas is None:
            savefig(savedir+'/'+saveas+'_%s.%s'%(t,extension))
        if not hook is None: hook(t)
        st=waitfor(st+1000/FPS)
    if not ff is None: 
        plt.figure(ff.number)

def phase_plane_animation_arraygrid(session,tr,fa=10,fb=45,\
    epoch=None,skip=1,saveas=None,hook=None,FPS=30,stabilize=True,markersize=1.5):
    '''
    phase_plane_animation(session,tr)
    '''
    warn('Also plots "bad" channels')
    print(session, tr)
    # save current figure so we can return to it
    ff=gcf()
    ax=gca()
    # get time base
    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]
    # retrieve filtered array data
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    # locate all pairs
    pairs = {}
    for a in areas:
        pairs[a] = get_all_pairs_ordered_as_channel_indecies(session,a)
    # compute phase velocities for stabilization
    alldata = concatenate(data.values(),1)
    phasedt = rewrap(diff(angle(alldata),1,0))
    phasev  = median(phasedt,axis=1)
    phaseshift = append(0,cumsum(phasev))
    # compute stabilization differently, using median phase
    #phaseshift = angle(mean(alldata,axis=1))
    # PREPARE FIGURE
    figtitle = 'Analytic Signal %s-%sHz\n%s trial %s'%(fa,fb,session,tr)
    if not saveas is None:
        saveas += '_%s_%s_%s_%s'%(session,tr,fa,fb)
    figure('Phase Plane')
    a2 = cla()
    # DETERMINE NICE SQUARE AXIS BOUNDS
    M = 10
    for a in areas:
        M = max(M,int(ceil(np.max(abs(data[a]))/10.))*10)
    complex_axis(M)
    title(figtitle+' t=%dms'%times[0])
    # prepare output directory if we're going to save this
    if not saveas is None:
        savedir = './'+saveas
        ensuredir(savedir)
    # prepare stuff for blitting
    aa = gca()
    canvas = aa.figure.canvas
    background = canvas.copy_from_bbox(aa.bbox)
    def updateline(time):
        print('!!! t=',time)
        canvas.restore_region(background)
        aa.draw_artist(line)
        aa.figure.canvas.blit(ax.bbox)
    # initialize points
    scat={}
    grid={}
    for i,a in en(areas):
        points = c2p(data[a][0])
        c = darkhues(9)[i*3+2]
        scat[a] = scatter(*points,s=markersize**2,color=c,label=a)
        lines = []
        for ix1,ix2 in pairs[a]:
            p1=points[:,ix1]
            p2=points[:,ix2]
            line = plot([p1[0],p2[0]],[p1[1],p2[1]],color=c,lw=0.4)[0]
            lines.append((line,ix1,ix2))
        grid[a]=lines
    nice_legend()
    # perform animation
    st = now()
    for i,t in en|times:
        stabilizer = exp(-1j*phaseshift[i]) if stabilize else 1
        # update via blitting instead of draw
        canvas.restore_region(background)
        for a in areas:
            points = c2p(stabilizer*data[a][i])
            scat[a].set_offsets(points.T)
            for line,ix1,ix2 in grid[a]:
                p1=points[:,ix1]
                p2=points[:,ix2]
                line.set_data([p1[0],p2[0]],[p1[1],p2[1]])
                aa.draw_artist(line)
        title(figtitle+' t=%sms'%t)
        # update via blitting instead of draw
        aa.figure.canvas.blit(ax.bbox)
        # draw()
        if not saveas is None:
            savefig(savedir+'/'+saveas+'_%s.png'%t)
        if not hook is None: hook(t)
        st=waitfor(st+1000/FPS)
    if not ff is None: figure(ff.number)

def phase_animation(phases,skip=1,saveas=None,hook=None,FPS=15,markersize=1.5):
    '''
    phase_animation(session,tr)
    '''
    assert 0 # not implemented
    # save current figure so we can return to it
    ff=gcf()
    ax=gca()
    # get time base
    times = arange(shape(phases)[1])
    # PREPARE FIGURE
    figtitle = 'Analytic Signal'
    figure('Phase Plane')
    a2 = cla()
    M = int(ceil(np.max(abs(phases))))
    complex_axis(M)
    title(figtitle + ' t=%dms'%times[0])
    # initialize points
    scat= scatter(*c2p(data[a][0]),s=markersize**2,color='k',label=a)
    nice_legend()
    # perform animation
    st = now()
    for i,t in en|times:
        print(i,t)
        for a in areas:
            scat[a].set_offsets(c2p(data[a][i]))
        title(figtitle+' t=%sms'%t)
        draw()
        if not saveas is None:
            savefig(savedir+'/'+saveas+'_%s.png'%t)
        if not hook is None: hook(t)
        st=waitfor(st+1000/FPS)
    if not ff is None: figure(ff.number)

def phase_plane_animation_distribution(session,tr,areas,fa=10,fb=45,epoch=None,\
    skip=1,saveas=None,hook=None,FPS=30,stabilize=False,M=None,extension='png',markersize=1.5):
    areacolors = [OCHRE,AZURE,RUST]
    '''
    Test code:
    from os.path import expanduser
    session = 'SPK120918'
    area = 'M1'
    trial = 2
    tr = trial
    epoch = None
    fa,fb = 15,30
    fa = int(round(fa))
    fb = int(round(fb))
    close('all')
    phase_plane_animation_distribution(session,tr,['M1','PMv','PMd'],fa=10,fb=45,FPS=Inf,
    M=100,skip=1,extension='pdf',saveas='cgauss')
    '''
    models = [logpolar_stats,complex_gaussian]
    modelcolors = ['c','k']
    modellw = 1.5
    if not saveas is None:
        saveas += '_'+'_'.join(map(str,areas))
    '''
    phase_plane_animation(session,tr)
    '''
    print(session, tr)
    # save current figure so we can return to it
    ff=gcf()
    ax=gca()
    # get time base
    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]
    # retrieve filtered array data
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    # compute phase velocities for stabilization
    alldata = concatenate(data.values(),1)
    #phasedt = rewrap(diff(angle(alldata),1,0))
    #phasev  = median(phasedt,axis=1)
    #phaseshift = append(0,cumsum(phasev))
    # compute stabilization differently, using median phase
    phaseshift = angle(mean(alldata,axis=1))
    # PREPARE FIGURE
    figtitle = 'Analytic Signal %s-%sHz\n%s trial %s'%(fa,fb,session,tr)
    if not saveas is None:
        saveas += '_%s_%s_%s_%s'%(session,tr,fa,fb)
    figure('Phase Plane',figsize=(4,4))
    a2 = cla()
    if M is None:
        M = 10
        for a in areas:
            M = max(M,int(ceil(np.max(abs(data[a]))/10.))*10)
    complex_axis(M)
    title(figtitle+' t=%dms'%times[0])
    tight_layout()
    # prepare output directory if we're going to save this
    if not saveas is None:
        savedir = './'+saveas
        ensuredir(savedir)
    # initialize points and lines
    scat={}
    frame = []
    for i,a in en(areas):
        x = data[a][0]
        scat[a] = scatter(*c2p(x),s=markersize**2,color=areacolors[i],label=a)
        frame.extend(x)
    frame = arr(frame)
    model = [m(frame) for m in models]
    modellines = []
    for i,m in en|model:
        distcolor = modelcolors[i]
        lines = [plot(*c2p(a),color=distcolor,lw=modellw,zorder=Inf)[0] for a in m]
        modellines.append(lines)
    nice_legend()
    # perform animation
    st = now()
    for i,t in en|times:
        stabilizer = exp(-1j*phaseshift[i]) if stabilize else 1
        frame = []
        for a in areas:
            x = stabilizer*data[a][i]
            scat[a].set_offsets(c2p(x).T)
            frame.extend(x)
        frame = arr(frame)
        for i,(lines,m) in en|iz(modellines,[m(frame) for m in models]):
            for l,x in iz(lines,m):
                l.set_data(*c2p(x))
        title(figtitle+' t=%sms'%t)
        draw()
        if not saveas is None:
            savefig(savedir+'/'+saveas+'_%s.%s'%(t,extension))
        if not hook is None: hook(t)
        st=waitfor(st+1000/FPS)
    if not ff is None: figure(ff.number)



if __name__=='__main__':

    from cgid.setup import *
    close('all')
    figure('Phase Plane',figsize=(4,4))
    phase_plane_animation('SPK120918',2,areas=['M1','PMv','PMd'],fa=10,fb=45,epoch=(6,120-1000,190-1000),skip=10,saveas='Chapter4Figure10',hook=None,FPS=2,stabilize=False,extension='pdf',markersize=2,figtitle='')


    """
    session = 'SPK120918'
    area = 'M1'
    trial = 16
    tr = trial
    epoch = None
    fa,fb = cgid.beta.estimate_beta_band(session,None,bw=7,doplot=1)
    fa = int(round(fa))
    fb = int(round(fb))
    close('all')

    # pretty animation but it runs forever, skip for now
    #phase_plane_animation(session,2,areas)

    # retrieve filtered array data
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb)
        data[a]=x.T

    t = 148

    frame = []
    for a in areas:
        frame.extend(data[a][t,:])

    frame = arr(frame)

    log(abs(frame)),angle(frame)

    # set to zero mean phase
    theta    = angle(mean(frame))
    rephased = frame*exp(1j*-theta)

    # convert to log-polar
    x = log(abs(rephased))
    y = angle(rephased)

    # use 2D gaussian approximation
    mx = mean(x)
    my = mean(y)
    cx = x - mx
    cy = y - my
    cm = cov(cx,cy)
    print(mx,my,cm)
    w,v = eig(cm)
    v = v[0,:]+1j*v[1,:]
    origin = mx + 1j*my
    axis1 = origin + v[0]*w[0]*linspace(-1,1,100)
    axis2 = origin + v[1]*w[1]*linspace(-1,1,100)
    circle = exp(1j*linspace(0,2*pi,100))
    circle = p2c(dot(cm,[real(circle),imag(circle)]))+origin

    clf()
    scatter(x,y)
    plot(*c2p(axis1),color='r',lw=1,zorder=Inf)
    plot(*c2p(axis2),color='r',lw=1,zorder=Inf)
    plot(*c2p(circle),color='r',lw=1,zorder=Inf)

    # send back to original coordinates
    clf()
    scatter(*c2p(frame))
    phase = exp(1j*theta)
    plot(*c2p(exp(axis1)*phase),color='r',lw=2,zorder=Inf)
    plot(*c2p(exp(axis2)*phase),color='r',lw=2,zorder=Inf)
    plot(*c2p(exp(circle)*phase),color='r',lw=2,zorder=Inf)

    # nice animations but take a long time to run
    #phase_plane_animation_distribution(session,tr,areas,fa=17,fb=27,FPS=15,M=70)
    #phase_plane_animation_distribution(session,tr,areas,fa=10,fb=45,FPS=Inf,
    #    M=70,skip=2,extension='pdf',saveas='allmethods')

    skip = 10

    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]

    # retrieve filtered array data
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    ccs=[]
    for i,t in en|times:
        frame = []
        for a in areas:
            x = data[a][i]
            frame.extend(x)
        frame = arr(frame)
        concent = sqrt(-2*log(abs(mean(frame))/mean(abs(frame))))* mean(abs(frame))
        print(i,t,concent)
        ccs.append(concent)
    ccs = arr(ccs)
    '''
    i=4322
    frame = []
    for a in areas:
        x = stabilizer*data[a][i]
        frame.extend(x)
    frame = arr(frame)
    complex_gaussian(frame,1)
    logpolar_stats(frame,1)
    '''
    frame = (5+randn(2000))*exp(1j*2*pi*rand(2000))
    complex_gaussian(frame,1)
    logpolar_stats(frame,1)
    scatter(*c2p(frame))
    models = [logpolar_stats,complex_gaussian]
    modelcolors = [TURQUOISE,BLACK]
    modellw = 1.5
    figtitle = 'Simulated data'
    figure(figsize=(4,4))
    a2 = cla()
    complex_axis(10)
    title(figtitle)
    tight_layout()
    model = [m(frame) for m in models]
    modellines = []
    for i,m in en|model:
        distcolor = modelcolors[i]
        lines = [plot(*c2p(a),color=distcolor,lw=modellw,zorder=Inf)[0] for a in m]
        modellines.append(lines)
    scatter(*c2p(frame),color='k',zorder=-10,s=2**2)
    savefig('sumulated.pdf')
    frame = (5*randn(2000))*exp(1j*2*pi*rand(2000))
    complex_gaussian(frame,1)
    logpolar_stats(frame,1)
    scatter(*c2p(frame))
    models = [logpolar_stats,complex_gaussian]
    modelcolors = ['c','k']
    modellw = 1.5
    figtitle = 'Simulated data'
    figure(figsize=(4,4))
    a2 = cla()
    M=10
    complex_axis(M)
    title(figtitle)
    tight_layout()
    force_aspect()
    model = [m(frame) for m in models]
    modellines = []
    for i,m in en|model:
        distcolor = modelcolors[i]
        lines = [plot(*c2p(a),color=distcolor,lw=modellw,zorder=Inf)[0] for a in m]
        modellines.append(lines)
    scatter(*c2p(frame),color='k',zorder=-10,s=2**2)
    savefig('sumulated2.pdf')

    '''
    # negative polar exploration -- is strange, do not use
    close('all')
    session = 'SPK120918'
    tr = 2
    fa = 10
    fb = 45
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    #i=423
    i=119
    frame = []
    for a in areas:
        x = data[a][i]
        frame.extend(x)
    frame = arr(frame)
    # rotate just because
    frame = frame*exp(1j)
    models = [logpolar_stats,complex_gaussian]
    modelcolors = [TURQUOISE,BLACK]
    modellw = 1.5
    figure(1,figsize=(8,8))
    clf()
    a2 = cla()
    complex_axis(70)
    title(figtitle)
    tight_layout()
    force_aspect()
    scatter(*c2p(frame),color='k',zorder=-10,s=2**2)
    model = [m(frame,False) for m in models]
    modellines = []
    for i,m in en|model:
        distcolor = modelcolors[i]
        lines = [plot(*c2p(a),color=distcolor,lw=modellw,zorder=Inf)[0] for a in m]
        modellines.append(lines)
    abspolar_stats(frame,doplot=1)
    '''
    """

    # quadratic exploration -- is strange, do not use
    """
    close('all')
    session = 'SPK120918'
    tr = 2
    fa = 10
    fb = 45
    data = {}
    for a in areas:
        print('loading area',a)
        x = get_all_analytic_lfp(session,a,tr,epoch,fa,fb,onlygood=True)[:,::skip]
        data[a]=x.T
    #i=423
    i=119
    frame = []
    for a in areas:
        x = data[a][i]
        frame.extend(x)
    frame = arr(frame)
    # rotate just because
    frame = frame*exp(1j)
    models = [logpolar_stats,complex_gaussian]
    modelcolors = [TURQUOISE,BLACK]
    modellw = 1.5
    figure(1,figsize=(8,8))
    clf()
    a2 = cla()
    complex_axis(70)
    title(figtitle)
    tight_layout()
    force_aspect()
    scatter(*c2p(frame),color='k',zorder=-10,s=2**2)
    model = [m(frame,False) for m in models]
    modellines = []
    for i,m in en|model:
        distcolor = modelcolors[i]
        lines = [plot(*c2p(a),color=distcolor,lw=modellw,zorder=Inf)[0] for a in m]
        modellines.append(lines)
    abspolar_stats(frame,doplot=1)
    z = arr(frame)
    z = z**2
    weights = ones(shape(z))
    weights = weights/sum(weights)
    x = real(z)
    y = imag(z)
    # use 2D gaussian approximation
    mx = dot(weights,x)
    my = dot(weights,y)
    cx = x - mx
    cy = y - my
    #cm = cov(cx,cy)
    correction = sum(weights)/(sum(weights)**2-sum(weights**2))
    cxx = dot(weights,cx*cx)*correction
    cxy = dot(weights,cx*cy)*correction
    cyy = dot(weights,cy*cy)*correction
    cm = arr([[cxx,cxy],[cxy,cyy]])
    sm = cholesky(cm)
    w,v = eig(cm)
    v = v[0,:]+1j*v[1,:]
    origin = mx + 1j*my
    w = sqrt(w)
    axis1 = origin + v[0]*w[0]*linspace(-1,1,100)
    axis2 = origin + v[1]*w[1]*linspace(-1,1,100)
    circle = exp(1j*(linspace(0,2*pi,100)))
    circle = p2c(dot(sm,[real(circle),imag(circle)]))+origin
    #scatter(*c2p(z))
    a11 = sqrt(axis1)
    a21 = sqrt(axis2)
    c1  = sqrt(circle)
    a12 = -sqrt(axis1)
    a22 = -sqrt(axis2)
    c2  = -sqrt(circle)
    plot(*c2p(a11),color='r',lw=2,zorder=Inf)
    plot(*c2p(a21),color='r',lw=2,zorder=Inf)
    plot(*c2p(c1),color='r',lw=2,zorder=Inf)
    plot(*c2p(a12),color='r',lw=2,zorder=Inf)
    plot(*c2p(a22),color='r',lw=2,zorder=Inf)
    plot(*c2p(c2),color='r',lw=2,zorder=Inf)
    """




