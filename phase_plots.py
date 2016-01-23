'''
Utility functions for making nicely upsampled plots of phase delays and
phase gradients for the 10x10 Utah arrays.
'''

from neurotools.color import *
from matplotlib.pyplot import *
from neurotools.nlab import *

def array_imshow(data,vmin=None,vmax=None,cmap=extended,origin='lower',
    drawlines=1,interpolation='bicubic',extent=(0,4,0,4),ctitle=''):
    if extent!=(0,4,0,4):
        print 'different size?'
    if vmin is None: 
        vmin=np.nanmin(data)
        print 'vmin=',vmin
    if vmax is None: 
        vmax=np.nanmax(data)
        print 'vmax=',vmax
    #data[where(M<=0)]=NaN
    imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,origin=origin,
        interpolation=interpolation,extent=extent)
    #crossbad(M)
    if drawlines:
        for i in linspace(0,10,11):
            axvline(i*4.0/10,color='w',lw=0.3)
            axhline(i*4.0/10,color='w',lw=0.3)
    xlim(0,4)
    ylim(0,4)
    nicex()
    nicey()
    xlabel('mm')
    ylabel('mm')
    fudgex()
    fudgey(3)
    draw()
    cax=good_colorbar(vmin,vmax,cmap,ctitle,sideways=1)
    fudgey(4,cax)
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
    '''
    upsampled = dct_upsample(data,factor)
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = shape(upsampled)
    fw,fh  = float32(shape(data)[:2])*factor
    dw,dh  = (fw-sw)/fw*0.5*4,(fh-sh)/fw*0.5*4
    extent = (dw,dw+sw/fw*4,dh,dh+sh/fh*4)
    print 'extent=',extent
    if vmin is None: 
        vmin=np.nanmin(data)
        print 'vmin=',vmin
    if vmax is None: 
        vmax=np.nanmax(data)
        print 'vmax=',vmax
    #data[where(M<=0)]=NaN
    imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,origin=origin,
        interpolation=interpolation,extent=extent)
    #crossbad(M)
    if drawlines:
        for i in linspace(0,10,11):
            axvline(i*4.0/10,color='w',lw=0.3)
            axhline(i*4.0/10,color='w',lw=0.3)
    xlim(0,4)
    ylim(0,4)
    nicex()
    nicey()
    xlabel('mm')
    ylabel('mm')
    fudgex()
    fudgey(3)
    draw()
    cax=good_colorbar(vmin,vmax,cmap,ctitle,sideways=1)
    fudgey(4,cax)
    return cax


def overlay_gradient(phase_gradient):
    NH,NW = shape(phase_gradient)[:2]
    p = arange(max(NW,NH))+0.5
    for row in range(NH):
        py = p[row]
        for col in range(NW):
            dz = phase_gradient[row,col]*0.25
            a = dz*.7*exp( .5j);
            b = dz*.7*exp(-.5j);
            px = p[col]
            plot([px,px+real(dz)],[py,py+imag(dz)],color='k')[0]
            plot([px,px-real(dz)],[py,py-imag(dz)],color='k')[0]
            plot([px-a.real,px-real(dz),px-b.real],
                 [py-a.imag,py-imag(dz),py-b.imag],color='k')[0]
    gca().tick_params(axis=u'both', which=u'both',length=0)


def phase_delay_plot(mean_analytic_signal,cm=isolum,UPSAMPLE=50,smooth=2.3,NLINE=6):
    '''
    Accepts an analytic signal map, upsamples it, and plots in the current
    axis the phases. For now, expects a 10x10 array 4x4mm is size. 
    '''
    cla()
    upsampled = dct_upsample(dct_cut_antialias(mean_analytic_signal,smooth),UPSAMPLE)
    # upsampling trims the array a bit
    # figure out how to re-center the trimmed array:
    sw,sh  = shape(upsampled)
    fw,fh  = float32(shape(mean_analytic_signal)[:2])*UPSAMPLE
    dw,dh  = (fw-sw)/fw*0.5*4,(fh-sh)/fw*0.5*4
    extent = (dw,dw+sw/fw*4,dh,dh+sh/fh*4)
    # extract angles
    amean  = angle(mean(upsampled))
    angles = (angle(upsampled*exp(-1j*amean))+2*pi+pi+1.75)%(2*pi)
    # plot phase angles
    cax    = array_imshow(angles,0,2*pi,cm,ctitle='',extent=extent)
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
    xlabel('mm',fontsize=11)
    sca(oldax)
    return cax


def vector_summary_plot_subroutine(mu,sigma,vectors):
    '''
    Plotting subroutine. Takes order parameter R and a list of vectors
    that has been standardiszed.
    
    TODO: circular gaussian plot is a little off here
    '''
    cla()
    limit = 2.0
    
    # draw the vectors
    for v in vectors:
        arrow(0,0,v.real,v.imag,head_width=0.05,head_length=0.1,fc='k',color='k',lw=.2)
        
    # draw some axis lines
    plot([-limit,limit],[0,0],lw=2,color='k')
    plot([0,0],[-limit,limit],lw=2,color='k')
    xlim(-limit,limit)
    ylim(-limit,limit)
    
    # plot the phase distribution as if circular gaussian (?correct?)
    h = linspace(-pi,pi,361)
    p = 1.2
    plot(cos(h+mu)*p,sin(h+mu)*p,lw=0.5,color='k')
    p = npdf(0,sigma,h)*0.4 + 1.2
    plot(cos(h+mu)*p,sin(h+mu)*p,lw=0.5,color='k')
    
    # clean up the plot
    force_aspect(1)
    draw()
    noaxis()
    nox()
    noy()
    
    # draw a mean vector
    mv = mean(vectors)
    arrow(0,0,mv.real,mv.imag,head_width=0.05,head_length=0.1,
        fc='w',color='w',lw=5,zorder=1e9)
    arrow(0,0,mv.real,mv.imag,head_width=0.05,head_length=0.1,
        fc='r',color='r',lw=3,zorder=1e10)
        
    # draw annotation for circular standard deviation
    #plot([1.2,1.2],[0,sigma],lw=1,color='k')
    #text(1.2+pixels_to_xunits(3),sigma/2,'$S$',
    #    horizontalalignment='left',
    #    verticalalignment='center',
    #    fontsize=16)

def vector_summary_plot(vectors):
    '''
    Does a summary plot of the phases of the vectors provided.
    vectors should be an array of complex numners
    '''
    cla()
    vectors = vectors/abs(vectors)
    R  = abs(mean(vectors))
    S  = sqrt(-2*log(R))
    mu = angle(mean(vectors))
    vector_summary_plot_subroutine(mu,S,vectors)

def weighted_vector_summary_plot(vectors):
    '''
    Does a summary plot of the phases of the vectors provided.
    vectors should be an array of complex numners
    Does a weighted average of direction by the vector magnitudes.
    CAUTION: Normalized vectors by squared mag for plotting, 
    need to check that this is OK
    '''
    cla()
    R  = abs(mean(vectors))/mean(abs(vectors))
    S  = sqrt(-2*log(R))
    mu = angle(mean(vectors))
    vectors = vectors/mean(abs(vectors)**2)**.5
    vector_summary_plot_subroutine(mu,S,vectors)






