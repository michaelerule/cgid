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

def heuristic_solver_planar((i,xys,times,data)):
    objective = makeLSQminimizerPlane(xys,times,real(data))
    result = leastsq(objective,heuristic_B_planar(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)

def array_single_frame_linear_wave_model(frame):
    pass
    



