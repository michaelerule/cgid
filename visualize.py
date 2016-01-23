'''
Note: Array wire bundle exits to the right
'''

from neurotools.plot   import plot_complex
from matplotlib.pyplot import gca, gcf, figure, clf
from cgid.data_loader  import get_trial_times_ms
from cgid.lfp          import *
from cgid.array_mapper import *
from neurotools.nlab   import *
from matplotlib.pyplot import *

def event_preview(session,area,event,upsample=6,cut=2):
    t,f,s = event
    z = packArrayDataInterpolate(session,area,s)
    for i,t in en(t):
        j = z[:,:,i]
        j = dct_cut_antialias(j,cut)
        j = dct_upsample(j,6)
        plot_complex(j)

def activity_preview(session,area,s,upsample=6,cut=1):
    nch,t = shape(s)
    z = packArrayDataInterpolate(session,area,s)
    for i in range(t):
        j = z[:,:,i]
        j = dct_cut_antialias(j,cut)
        j = dct_upsample(j,upsample)
        plot_complex(j)

def play_session(session,area,fa=10,fb=45,skip=5,upsample=6,cut=1):
    for tr in get_good_trials(session,area):
        #def get_all_analytic_lfp(session,area,tr,fa,fb,epoch,onlygood)
        activity = get_all_analytic_lfp(session,area,tr,fa,fb,None,None)
        activity_preview(session,area,activity[:,::skip],upsample,cut)

def spiking_preview(session,area,data):
    pass

# keep as floats
anatomy_x_start = -4.
anatomy_w       = 17.
anatomy_y_start = -7.
anatomy_h       = 17.

def create_multiunit_canvas(
    canvas_w_px,
    canvas_h_px,
    session,
    shapes,
    starts,
    areas=None):
    '''
    Creates an image buffer for rendering array data to screen in 
    anatomical coordinates.
    
    This code is really complicated.
    '''
    if areas is None: areas=['M1','PMv','PMd']
    # prepare textures
    canvas = zeros((canvas_w_px,canvas_h_px),dtype=int32)
    # anatomical coordinates are in mm, relative to the M1 array
    anatomy = {}
    for a in areas:
        # retrieve the geometry information for this array
        corners,positions = getElectrodePositions(session,a)
        top_right, bottom_right, bottom_left, top_left = corners
        print 'corners',corners
        # use the anatomical quad coordinates to make a change of basis
        # matrix to convert from image to array 
        # note: as defined, the "x" direction in the arrays is the second axis
        # and the y direction is the first ( row major order )
        B1 = top_right   - top_left
        B2 = bottom_left - top_left
        basis = array([B1,B2])
        trans = inv(basis)
        # anatomical array quad
        anatomy_start = array([anatomy_x_start,anatomy_y_start])
        anatomy_scale = array([canvas_h_px/anatomy_h,canvas_w_px/anatomy_w])
        quad = (corners-anatomy_start)*anatomy_scale
        anatomy[a]=quad
        # iterate over the bounding box and convert pixel locations into array
        # coordinates. Skip the pixels that are out of bounds.
        # first convert pixel coordinates into anatomical
        # If the pixel is in-bounds, we need to re-scale it to the array
        # size
        box_bottom = max(0,int(np.min(quad[:,1])-1))
        box_top    = min(canvas_h_px-1,int(np.max(quad[:,1])-1))
        box_left   = max(0,int(np.min(quad[:,0])+1))
        box_right  = min(canvas_w_px-1,int(np.max(quad[:,0])+1))
        print 'image box:',box_bottom,box_top,box_left,box_right
        h,w = shapes[a]
        scale = max(w,h)  # update if ever have array truncated on both axes
        print a,'scale',scale
        for y in range(box_bottom-1,box_top+2):
            for x in range(box_left-1,box_right+2):
                # convert from image coordinates to anatomical coordinates
                ax = (x)*anatomy_w/canvas_w_px+anatomy_x_start
                ay = (y+.5)*anatomy_h/canvas_h_px+anatomy_y_start
                p = array([ax,ay])
                # change from anatomical coordinates to array coordinates
                q = dot(p-top_left,trans)
                ix,iy = q
                # skip out of bounds points
                if ix<0 or iy<0: continue
                # scale to data dimentions
                ix=int(((ix)*scale))
                iy=int(((iy)*scale))
                # skip out of bounds points
                if ix>=w:continue # some arrays are truncated on one axis
                if iy>=h:continue # don't try to draw there
                ir=ix+iy*w+starts[a]
                canvas[y,x]=ir
    return canvas,anatomy


def draw_array_outlines(
    session=None,
    anatomy=None,
    color='K',
    lw=1.5,
    areas=None,
    draw_wire=True):
    '''
    Draws outlines of the array locations in anatomical coordinates.
    
    Either session or anatomy must be not None.
    
    If session is specified (is not None), this will plot the array
    annotation in the current axis with units of mm
    
    If anatomy is specified (is not None), it should be a dictionary that
    maps array names to quadrilateral outlines for the array perimeter, in
    the order top_left, bottom_left, bottom_right, top_right
    '''
    assert (session==None)!=(anatomy==None)
    if areas is None: areas=['M1','PMv','PMd']
    for a in areas:
        # retrieve the geometry information for this array
        if not anatomy is None:
            quad = anatomy[a]
        else:
            quad,_ = getElectrodePositions(session,a)
        if draw_wire:
            # draw annotation for the wire bundle
            topleft, bottomleft, bottomright, topright = quad
            wy = 0.5*(bottomright+topright)
            wx = wy-0.5*(bottomleft-bottomright)
            tri = [bottomright,wx,topright]
            plot(*zip(*tri),color=color,
                lw=lw,solid_capstyle='round',zorder=inf)
        quad = list(quad)+[quad[0]]
        plot(*zip(*quad),color=color,
            lw=lw,solid_capstyle='round',zorder=inf)

def get_phase_gradients_for_animation(session,trial,
    areas=None,
    fa=10,
    fb=45,
    epoch=None,
    cut=1.7,
    skip=3):
    print 'COMPLETELY IGNORING SMOOTHING SCALE PARAMETER AND USING 1.7'
    print 'Also, normalizing the gradient vectors to unit length'
    if areas    is None: areas=['M1','PMv','PMd']
    gradients={}
    for area in areas:
        x=get_array_packed_lfp_analytic(session,area,trial,epoch,fa,fb)
        a=abs(x)
        a=0.25*(a[1:,1:]+a[:-1,1:]+a[1:,:-1]+a[:-1,:-1])
        y=dct_cut_antialias(x,cut)
        g=array_phase_gradient(y)
        g=g/abs(g)*a/40
        gradients[area]=g[...,::skip]
    return gradients

def plot_derivative_anatomical(
    gradient_data,
    (positions,dx,dy),
    lines=None):
    '''
    Plots the gradient vectors in the correct anatomical locations.
    Need to also change basis to get the correct orientation.
    '''
    assert shape(positions)[:2]==shape(gradient_data)
    Nrows,Ncols = shape(positions)[:2]
    nrows,ncols = shape(gradient_data)
    if lines is None: lines={}
    for col in range(ncols):
        for row in range(nrows):
            location = positions[row,col]
            gradient = gradient_data[row,col]
            delta = (gradient.real*dx + gradient.imag*dy)/2
            z1 = location + delta
            z2 = location - delta
            X = [z1[0],z2[0]]
            Y = [z1[1],z2[1]]
            if (row,col) in lines:
                lines[row,col].set_xdata(X)
                lines[row,col].set_ydata(Y)
            else:
                lines[row,col] = plot(X,Y,color='r')[0]
    return lines

def arrays_video_gradient(
    times,
    arraydata,
    phase_gradients,
    session,
    skip=3,
    figtitle=None,
    hook=None,
    FPS=20,
    areas=None,
    saveas=None,
    plot_gradient=True):
    '''
    Animation function. Units are millimeters. 
    
    Args:
        times : list of timepoints for each frame
        arraydata : array data to show. This needs to be a dictionary mapping
            area names to array-packed Nrows x Ncols lists of frames. The 
            data may be (is required to be?) complex.
        phase_gradients : Vector field info. Also a dictionary. 
        session : 
        skip=3 : only render every skip frames
        figtitle : Optionally, a figure title may be specified. Defaults to
            the name of the session.
        interp : 'bicubic',
        hook : None,
        FPS : Frames per second. Defaults to 20
        areas : None,
        canvas_N : None,
        saveas : None,
        plot_gradient : True
    '''
    # We can opt to plot only some areas, but by default we assume that 
    # all areas are being plotted. Note that the arraydata and the 
    # phase_gradients arguments must contain corresponding entries for 
    # each area being plotted.
    if areas    is None: areas=['M1','PMv','PMd']
    # Set figure title to the name of the session if not otherwise specified
    if figtitle is None: figtitle = '%s'%session
    # Prepare plot: axis labels, title, etc. 
    # Draw the axis to screen so we can measure how many pixels it uses
    # This lets us define an image the exact resolution of the canvas
    # and render to the axis as if it were just another image buffer
    clf()
    xlabel('Distance rostrally from M1 implant (mm)')
    ylabel('Distance dorsally from M1 implant (mm)')
    title(figtitle + ' t=%dms'%times[0])
    force_aspect()
    noaxis()
    tight_layout()
    draw()
    w,h = get_ax_size()
    canvas_N = int(ceil(max(w,h)))
    canvas_w_px = canvas_N
    canvas_h_px = canvas_N
    xticks([0,canvas_w_px-1],['%d'%anatomy_x_start,'%d'%(anatomy_x_start+anatomy_w)])
    yticks([0,canvas_h_px-1],['%d'%anatomy_y_start,'%d'%(anatomy_y_start+anatomy_h)])
    ylim(0,canvas_h_px-1)
    ylim(0,canvas_w_px-1)
    draw()
    # Get information about the areas being plotted. Array data is going
    # to be unwrapped and packed in a sequential array, so we need to 
    # store pointers to where each array stars in this buffer. Thus, we
    # need the length and shape of each bit of arraydata
    # Pack array data into the buffer for fast indexing by the video
    shapes = {}
    sizes  = {}
    data   = {}
    print "ERROR X IS FLIPPED"
    print "THERE IS A PROBLEM IN THE CODE"
    print "APPLYING PATCH BELOW BUT REALLY, YOU SHOULD LOCATE THIS BUG"
    for a in areas:
        x = arraydata[a][:,::-1,:]
        h,w,n = shape(x)
        shapes[a]=(h,w)
        sizes [a]=h*w
        data  [a]=reshape(x,(h*w,n))
    outofbounds  = ones((1,shape(data[a])[1]),dtype=data[a].dtype)
    video        = concatenate([outofbounds]+[data[a] for a in areas],0).T
    lengths      = arr([sizes[a] for a in areas])
    starts       = dict(zip(areas,cumsum([1,]+list(lengths))[:3]))
    # This creates a canvas which maps the packed arrayData into image
    # coodinates. Once a canvas is initialized, we can render do it
    # by using it as an index into our packed arrayData buffer.
    canvas,anatomy = create_multiunit_canvas(canvas_w_px,canvas_h_px,session,shapes,starts,areas=areas)
    canvas[canvas>=shape(video)[1]]=0 #???? WHY ????
    blank = canvas==0
    # outline arrays. This uses a modified anatomical coodinates
    draw_array_outlines(anatomy=anatomy)
    # prepare video
    video  = video.real
    video -= mean(video)
    video /= std(video)*1.5
    video  = (video+1)/2.
    video  = int32(255*video)
    video  = clip(video,0,255)
    # prepare plot: pre-render a frame, will update later
    # formerly used 
    # Now just plotting real component and overlaying phase field
    # Using the "extended" map (parula also available)
    RGBA = ones(shape(canvas)+(4,))
    img  = imshow(RGBA,interpolation='nearest',animated=True)
    # precompute the origins for phase gradient vector fields
    positions = {}
    print anatomy
    for a in areas:
        positions[a]=get_interelectrode_positions(session,a,anatomy[a])
    print positions
    if plot_gradient:
        lines ={a:plot_derivative_anatomical(phase_gradients[a][...,0],positions[a]) for a in areas}
    if not saveas is None:
        savedir = './'+saveas
        ensuredir(savedir)
    # render video
    for i,(t,frame) in en|iz(times,video):
        RGBA[...,:3] = extended_data[frame[canvas]]
        RGBA[blank,:]=1
        img.set_data(RGBA)
        title(figtitle+' t=%sms'%t)
        if plot_gradient:
            for a in areas:
                plot_derivative_anatomical(
                    phase_gradients[a][...,i],
                    positions[a],
                    lines=lines[a])
        draw()
        if not saveas is None: savefig(savedir+'/'+saveas+'_%s.png'%t)
        if not hook   is None: hook(t)

def full_analytic_lfp_video(session,tr,fa=10,fb=45,epoch=None,\
    upsample=None,cut=0.4,skip=3,canvas_N=None,saveas=None,\
    interp='nearest',hook=None,FPS=20,plot_gradient=True):
    '''
    TEST CODE
    >>> # play an animation 
    >>> session = 'RUS120518'
    >>> for tr in get_good_trials(session):
    >>>     full_analytic_lfp_video(session,tr,fa=18,fb=23)
    >>> # Render 1 trial from Rusty 18 in both broad and narrow band
    >>> close('all')
    >>> figure(figsize=(5,5))
    >>> tr = get_good_trials(session,area)[0]
    >>> lfp = get_all_lfp(session,area,tr)
    >>> full_analytic_lfp_video(session,tr,fa=10,fb=45,saveas='broadbeta')
    >>> full_analytic_lfp_video(session,tr,fa=15,fb=25,saveas='beta')
    >>> full_analytic_lfp_video(session,tr,fa=18,fb=22,saveas='narrowbeta')
    '''
    # load and upsample all data
    print 'epoch is',epoch
    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]
    phase_gradients = get_phase_gradients_for_animation(
        session,tr,fa=10,fb=45,epoch=epoch,cut=cut,skip=skip)
    arraydata = {}
    if upsample is None:
        print 'No upsampling specified. using heuristics'
        force_aspect()
        draw()
        mm_per_electrode = 0.4
        mm_per_win       = anatomy_w
        px_per_win       = int(ceil(max(*get_ax_size())))
        px_per_electrode = mm_per_electrode * px_per_win / mm_per_win
        upsample = int(ceil(px_per_electrode))+3
        print 'upsample is',upsample
    for a in areas:
        x = get_array_packed_lfp_analytic(session,a,tr,epoch,fa,fb)[...,::skip]
        x -= mean(x,-1)[:,:,None]
        x /= std(x,-1)[:,:,None]
        if not cut is None: x = dct_cut_antialias(x,cut)
        if upsample>1:      x = dct_upsample(x,upsample)
        arraydata[a]=x    
    figtitle = 'Analytic LFP activity %s-%sHz\n%s trial %s'%(fa,fb,session,tr)
    if not saveas is None:
        saveas += '_%s_%s_%s_%s'%(session,tr,fa,fb)
    arrays_video_gradient(
        times,
        arraydata,
        phase_gradients,
        session,
        skip=skip,
        figtitle=figtitle,
        hook=hook,
        FPS=FPS)

def lookat(session,trial,time=0,tafter=None,fa=10,fb=45,upsample=1,
    cut=0.4,FPS=20,timebar=False,plot_gradient=True,skip=1):
    '''
    Preview / visualize analytic signals over the array
    '''
    ff=gcf()
    ax=gca()
    line = None
    try:
        canvas = ax.figure.canvas
        background = canvas.copy_from_bbox(ax.bbox)
        if timebar:
            line = ax.axvline(time,color='r')
        if tafter is None: tafter=7000-time
        def updateline(time):
            if not timebar: return
            canvas.restore_region(background)
            line.set_xdata([time,time])
            ax.draw_artist(line)
            ax.figure.canvas.blit(ax.bbox)
        figure('Video Preview')
        full_analytic_lfp_video(session,trial,fa,fb,
            epoch=(6,-1000+time,-1000+time+tafter),
            skip=skip,
            hook=updateline,
            upsample=upsample,
            cut=cut,
            FPS=FPS,
            plot_gradient=plot_gradient)
    finally:
        if not line is None: line.remove()
    if not ff is None: figure(ff.number)


def full_MUA_lfp_video(session,tr,epoch=None,fc=250,fsmooth=30,upsample=1,
    cut=0.4,skip=3,canvas_N=None,saveas=None):
    ''' 
    Routine to visualize the multiunit activity. 
    >>> close('all')
    >>> figure(figsize=(6,6))
    >>> session = 'SPK120918'
    >>> session = 'RUS120518'
    >>> for tr in get_good_trials(session):
    >>>     full_MUA_lfp_video(session,tr,upsample=6,cut=0.1,fsmooth=30)
    '''
    times = get_trial_times_ms(session,'M1',tr,epoch)[::skip]
    arraydata = {}
    for a in areas:
        print 'loading area',a
        x = hilbert(zscore(
            get_all_MUA_lfp(session,a,tr,epoch,fc,fsmooth).T).T)[:,::skip]
        x = pack_array_data_interpolate(session,a,x)
        x = dct_cut_antialias(x,cut)
        if upsample>1: x = dctUpsample(x,upsample)
        arraydata[a]=x    
    figtitle = 'MUA LFP activity <%sHz\n%s trial %s'%(fc,session,tr)
    if not saveas is None:
        saveas += '_%s_%s_%s_%s'%(session,tr,fa,fb)
    arrays_video(times,arraydata,session,skip,canvas_N,saveas=saveas,
        figtitle=figtitle,interp='nearest')
    






