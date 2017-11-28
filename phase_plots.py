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

'''
Utility functions for making nicely upsampled plots of phase delays and
phase gradients for the 10x10 Utah arrays.
'''

from neurotools.graphics.color import parula,extended,isolum
from neurotools.graphics.plot import nox,noy,fudgex,fudgey,noaxis
import matplotlib.pyplot as plt
import numpy as np

def array_imshow(data,vmin=None,vmax=None,cmap=extended,origin='lower',
    drawlines=1,interpolation='bicubic',extent=(0,4,0,4),ctitle='',spacing=0,
    draw_colorbar=True):
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
            plt.axvline(i*4.0/10,color='w',lw=0.3)
            plt.axhline(i*4.0/10,color='w',lw=0.3)
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.nicexy()
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.fudgex()
    plt.fudgey(3)
    plt.draw()
    if draw_colorbar:
        cax=good_colorbar(vmin,vmax,cmap,ctitle,
            sideways=1,spacing=spacing)
        fudgey(4,cax)
    else:
        cax = None
    return cax

def array_plot_upsampled(data,factor,vmin=None,vmax=None,
    cmap=extended,
    origin='lower',
    drawlines=1,
    interpolation='bicubic',
    ctitle=''):
    '''
    Upsamples data by factor when plotting. Currently only set up for
    4x4mm 10x10 electrode MEAs.

    Handy because upsampling leaves strangely sized images, since you can
    only interpolate and not extrapolate -- and this hadles automatically
    placing the image at the correct location
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    upsampled = dct_upsample(data,factor)
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = np.shape(upsampled)
    fw,fh  = np.float32(np.shape(data)[:2])*factor
    dw,dh  = (fw-sw)/fw*0.5*4,(fh-sh)/fw*0.5*4
    extent = (dw,dw+sw/fw*4,dh,dh+sh/fh*4)
    print('extent=',extent)
    if vmin is None:
        vmin=np.nanmin(data)
        print('vmin=',vmin)
    if vmax is None:
        vmax=np.nanmax(data)
        print('vmax=',vmax)
    #data[where(M<=0)]=NaN
    imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,origin=origin,
        interpolation=interpolation,extent=extent)
    #crossbad(M)
    if drawlines:
        for i in linspace(0,10,11):
            plt.axvline(i*4.0/10,color='w',lw=0.3)
            plt.axhline(i*4.0/10,color='w',lw=0.3)
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.nicex()
    plt.nicey()
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.fudgex()
    plt.fudgey(3)
    plt.draw()
    cax=good_colorbar(vmin,vmax,cmap,ctitle,sideways=1)
    fudgey(4,cax)
    return cax

def overlay_gradient(phase_gradient):

    '''
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    NH,NW = shape(phase_gradient)[:2]
    p = arange(max(NW,NH))+0.5
    for row in range(NH):
        py = p[row]
        for col in range(NW):
            dz = phase_gradient[row,col]*0.25
            a = dz*.7*exp( .5j);
            b = dz*.7*exp(-.5j);
            px = p[col]
            plt.plot([px,px+real(dz)],[py,py+imag(dz)],color='k')[0]
            plt.plot([px,px-real(dz)],[py,py-imag(dz)],color='k')[0]
            plt.plot([px-a.real,px-real(dz),px-b.real],
                 [py-a.imag,py-imag(dz),py-b.imag],color='k')[0]
    plt.gca().tick_params(axis=u'both', which=u'both',length=0)

def get_upsampled_extent(upsampled_shape):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    raise NotImplementedError('This function isnt done yet')
    warn('assuming 10x10?')
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = np.shape(upsampled)
    fw,fh  = np.float32(np.shape(mean_analytic_signal)[:2])*UPSAMPLE
    dw,dh  = (fw-sw)/fw*0.5*4,(fh-sh)/fw*0.5*4
    extent = (dw,dw+sw/fw*4,dh,dh+sh/fh*4)

def phase_delay_plot(mean_analytic_signal,
    cm=isolum,UPSAMPLE=100,smooth=2.3,NLINE=6,
    recenter=True,draw_colorbar=True):
    '''
    Accepts an analytic signal map, upsamples it, and plots in the current
    axis the phases. For now, expects a 10x10 array 4x4mm is size.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    plt.cla()
    upsampled = dct_upsample(dct_cut_antialias(mean_analytic_signal,smooth),UPSAMPLE)
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = shape(upsampled)
    fw,fh  = float32(shape(mean_analytic_signal)[:2])*UPSAMPLE
    dw,dh  = (fw-sw)/fw*0.5*4,(fh-sh)/fw*0.5*4
    extent = (dw,dw+sw/fw*4,dh,dh+sh/fh*4)
    # extract angles
    if recenter:
        amean  = np.angle(np.mean(upsampled))
        angles = (np.angle(upsampled*np.exp(-1j*amean))+2*pi+pi+1.75)%(2*pi)
    else:
        angles = (np.angle(upsampled)+2*pi+pi+1.75)%(2*pi)
    # plot phase angles
    cax = array_imshow(angles,0,2*pi,
        cmap=cm,
        ctitle='',
        extent=extent,
        draw_colorbar=draw_colorbar)
    # add countours
    for phi in linspace(0,pi,NLINE+1)[:-1]:
        c=plt.contour((angles+phi)%(2*pi),[pi],linewidths=0.8,colors='k',extent=extent)
    # Add title
    title('Average phase delay',fontsize=13)
    # fix up colorbar axis labels
    oldax = plt.gca()
    if draw_colorbar:
        plt.sca(cax)
        plt.noaxis()
        cax.tick_params(axis=u'both', which=u'both',length=0,labelleft='off', labelright='on')
        plt.yticks([0,2*pi],['0','$2\pi$'])
        # redraw the plot
        plt.sca(oldax)
    plt.xlabel('mm',fontsize=11)
    plt.draw()
    return cax,extent


def vector_summary_plot_subroutine(mu,sigma,vectors):
    '''
    Plotting subroutine. Takes order parameter R and a list of vectors
    that has been standardiszed.

    TODO: circular gaussian plot is a little off here
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    plt.cla()
    limit = 2.0

    # draw the vectors
    for v in vectors:
        plt.arrow(0,0,v.real,v.imag,head_width=0.05,head_length=0.1,fc='k',color='k',lw=.2)

    # draw some axis lines
    plt.plot([-limit,limit],[0,0],lw=2,color='k')
    plt.plot([0,0],[-limit,limit],lw=2,color='k')
    plt.xlim(-limit,limit)
    plt.ylim(-limit,limit)

    # plot the phase distribution as if circular gaussian (?correct?)
    h = np.linspace(-pi,pi,361)
    p = 1.2
    plt.plot(np.cos(h+mu)*p,np.sin(h+mu)*p,lw=0.5,color='k')
    p = npdf(0,sigma,h)*0.4 + 1.2
    plt.plot(np.cos(h+mu)*p,np.sin(h+mu)*p,lw=0.5,color='k')

    # clean up the plot
    force_aspect(1)
    plt.draw()
    noaxis()
    nox()
    noy()

    # draw a mean vector
    mv = np.mean(vectors)
    plt.arrow(0,0,mv.real,mv.imag,head_width=0.05,head_length=0.1,
        fc='w',color='w',lw=5,zorder=1e9)
    plt.arrow(0,0,mv.real,mv.imag,head_width=0.05,head_length=0.1,
        fc='r',color='r',lw=3,zorder=1e10)

    # draw annotation for circular standard deviation
    #plt.plot([1.2,1.2],[0,sigma],lw=1,color='k')
    #plt.text(1.2+pixels_to_xunits(3),sigma/2,'$S$',
    #    horizontalalignment='left',
    #    verticalalignment='center',
    #    fontsize=16)

def vector_summary_plot(vectors):
    '''
    Does a summary plot of the phases of the vectors provided.
    vectors should be an array of complex numners
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    plt.cla()
    vectors = vectors/np.abs(vectors)
    R  = np.abs(np.mean(vectors))
    S  = np.sqrt(-2*np.log(R))
    mu = np.angle(np.mean(vectors))
    vector_summary_plot_subroutine(mu,S,vectors)

def weighted_vector_summary_plot(vectors):
    '''
    Does a summary plot of the phases of the vectors provided.
    vectors should be an array of complex numners
    Does a weighted average of direction by the vector magnitudes.
    CAUTION: Normalized vectors by squared mag for plotting,
    need to check that this is OK
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    plt.cla()
    R  = np.abs(np.mean(vectors))/np.mean(np.abs(vectors))
    S  = np.sqrt(-2*np.log(R))
    mu = np.angle(np.mean(vectors))
    vectors = vectors/np.mean(np.abs(vectors)**2)**.5
    vector_summary_plot_subroutine(mu,S,vectors)






