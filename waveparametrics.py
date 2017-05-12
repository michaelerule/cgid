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


'''
Things that are difficult to track. 
Parametric model based statistics.
'''

from cgid.phasetools import *

def makeLSQminimizerPlane(xy,time,neuraldata):
    nxy  = shape(xy)[0]
    nt   = shape(time)[0]
    time -= mean(time)
    xy   -= mean(xy,0)
    window = hanning(nt+2)[1:-1]
    def getResiduals(params):
        A,B,a,b,w = params
        residuals = zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                phase = a*x+b*y-w*t
                prediction = A*sin(phase)+B*cos(phase)
                residuals[ixy,it] = abs(neuraldata[it,ixy] - prediction)*window[it]
        return ravel(residuals)
    return getResiduals

def heuristic_solver_planar(args):
    (i,xys,times,data) = args
    objective = makeLSQminimizerPlane(xys,times,real(data))
    result = leastsq(objective,heuristic_B_planar(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)

def array_single_frame_linear_wave_model(frame):
    pass
    



