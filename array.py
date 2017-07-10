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

import cgid.lfp 
import cgid.tools
from neurotools.spatial.array import array_count_critical
from neurotools.tools import warn

DEFAULT_SPATIAL_SCALE_CUTOFF = 2.0

def get_analytic_frames(session,area,trial,epoch=None,fa=10,fb=45):
    '''
    Scripts to pack data into frames are in the visualze file.
    I've moved them to cgidtools.
    packArrayDataInterpolate(session,area,data)
    accept NChannel x Ntimes and returns frame-packed data w. interp'd electrodes
    elaborate on this, 
    '''
    if epoch is None: epoch = 6,-1000,6000
    lfp = cgid.lfp.get_all_analytic_lfp(session,area,trial,epoch,fa,fb,onlygood=False)
    return cgid.tools.pack_array_data_interpolate(session,area,lfp)

def data_count_critical(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF):
    frames = get_analytic_frames(session,area,trial,epoch,fa,fb)
    return array_count_critical(frames,upsample,cut,cutoff)

def data_count_clockwise(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)[0]
    return bandfilter(res,fb=fsmooth)

def data_count_widersyns(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)[1]
    return bandfilter(res,fb=fsmooth)

def data_count_singular(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)
    res = res[0]+res[1]
    return bandfilter(res,fb=fsmooth)

def data_count_winding(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)
    res = res[0]-res[1]
    return bandfilter(res,fb=fsmooth)

def data_count_saddles(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)[2]
    return bandfilter(res,fb=fsmooth)

def data_count_maxima(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)[3]
    return bandfilter(res,fb=fsmooth)

def data_count_minima(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)[4]
    return bandfilter(res,fb=fsmooth)

def data_count_extrema(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)
    res = res[3]+res[4]
    return bandfilter(res,fb=fsmooth)

def data_count_inflection(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res= data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)
    res = res[3]+res[4]+res[2]
    return bandfilter(res,fb=fsmooth)

def data_count_critical_combine(session,area,trial,epoch,fa,fb,upsample=3,cut=True,cutoff=DEFAULT_SPATIAL_SCALE_CUTOFF,fsmooth=5):
    res = data_count_critical(session,area,trial,epoch,fa,fb,upsample,cut,cutoff)
    res = sum(arr(res),0)
    return bandfilter(res,fb=fsmooth)
