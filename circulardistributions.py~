'''
Explore log-polar gaussian distributions for representing analytic
signal across the population.
'''

from matplotlib.pyplot import *
from neurotools.plot import *
from neurotools.color import *

def logpolar_gaussian(frame,doplot=False):
    # set to zero mean phase
    theta    = angle(mean(frame))
    rephased = frame*exp(1j*-theta)
    weights = abs(rephased)
    weights = weights/sum(weights)
    x = log(abs(rephased))
    y = angle(rephased)/4
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
    circle = exp(1j*linspace(0,2*pi,100))
    circle = p2c(dot(sm,[real(circle),imag(circle)]))+origin
    phase = exp(1j*theta)
    if doplot:
        plot(*c2p(exp(axis1)*phase),color='r',lw=2,zorder=Inf)
        plot(*c2p(exp(axis2)*phase),color='r',lw=2,zorder=Inf)
        plot(*c2p(exp(circle)*phase),color='r',lw=2,zorder=Inf)
    return exp(axis1)*phase,exp(axis2)*phase,exp(circle)*phase
    

def complex_gaussian(frame,doplot=False):
    # set to zero mean phase
    rephased = frame#*exp(1j*-theta)
    weights = ones(shape(rephased))
    weights = weights/sum(weights)
    # convert to log-polar
    x = real(rephased)
    y = imag(rephased)
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
    circle = exp(1j*linspace(0,2*pi,100))
    circle = p2c(dot(sm,[real(circle),imag(circle)]))+origin
    if doplot:
        plot(*c2p(axis1),color='r',lw=2,zorder=Inf)
        plot(*c2p(axis2),color='r',lw=2,zorder=Inf)
        plot(*c2p(circle),color='r',lw=2,zorder=Inf)
    return axis1,axis2,circle

def logpolar_stats(frame,doplot=False):
    z = mean(frame)
    r = mean(abs(frame))
    rl = mean(log(abs(frame)))
    rs = std(abs(frame))
    rsl = std(log(abs(frame)))
    w = frame / abs(frame)
    x = mean(w)
    theta = angle(x)
    #R = abs(x)
    R = abs(z) / r
    sd = sqrt(-2*log(R))
    print 'R,sd',R,sd
    cv = 1-R
    s = exp(rl)*exp(1j*theta)
    arc = exp(rl+theta*1j)*exp(1j*linspace(-sd,sd,100))
    circle = exp(1j*linspace(0,2*pi,100))
    circle = real(circle)*rsl + 1j*imag(circle)*sd
    circle = circle+rl+1j*theta
    circle = exp(circle)
    radial = arr([s*exp(-rsl),s*exp(rsl)])
    if doplot:
        plot(*c2p(circle),color='m',lw=2)
        plot(*c2p(arc),color='m',lw=2)
        plot(*c2p(radial),color='m',lw=2)
    return circle,arc,radial

def abspolar_stats(frame,doplot=False):
    z = frame
    phi = angle(mean(z**2))/2
    flip = sign(cos(angle(z)-phi))
    r = abs(z)*flip
    h = angle(z) + pi*int32(flip==-1)
    mr = mean(r)
    sr = std(r)
    mt = phi
    #s  = r*exp(1j*h)
    #st = sqrt(-2*log(abs(mean(s))/mean(abs(s))))
    st = sqrt(-2*log(abs(mean(exp(1j*h)))))
    arc    = mr*exp(1j*(phi+linspace(-st,st,100)))
    circle = exp(1j*linspace(0,2*pi,100))
    circle = (real(circle)*sr+mr)*exp(1j*(imag(circle)*st+phi))
    radial = arr([(mr-sr)*exp(1j*phi),(mr+sr)*exp(1j*phi)])
    if doplot:
        clf()
        plot(*c2p(circle),color='m',lw=2)
        plot(*c2p(arc),color='m',lw=2)
        plot(*c2p(radial),color='m',lw=2)
        scatter(*c2p([mr*exp(1j*phi)]),color='k',s=5**2)
    return circle,arc,radial



