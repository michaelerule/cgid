'''
Chapter 4 has this passage in it

Wavelengths ranged from ?? mm to ?? mm, with a median of ?? mm. 
High beta power (amplitude envelope >1.5 standard deviations ($\sigma$) of 
the beta band signal) was associated with waves about ?? and low beta power
(<1.5$\sigma$) associated with waves about ?? %MISSING DATA!%

Wavelength are only analyzed in area M1 as that is where the most planar
waves are seen. So we only need to do this for M1 monkey R and S

See if you can locate a figure called
plane_wavelength_examples_nointerp.pdf

The code that generated this should also be useful here

See these scripts for reference:
CGID_manuscript_3/20151007_wavelength_variation_get_examples.py
CGID_manuscript_3/20151014_wavelength_variation_get_examples.py
'''


print 'Running...'
try:
    cgid
    neurotools
except:
    import cgid, neurotools
    from cgid.setup import *
    from neurotools.nlab import *
    from cgid.plotting_helper_functions import array_imshow, phase_delay_plot

Fs           = 1000
bandwidth    = 5.0
SKIP         = 50
session,area = 'SPK120925','M1'
todo = [('RUS120518', 'M1'),
        ('SPK120925', 'M1'),]
SCUT  = pi/4
RCUT  = exp(SCUT**2/-2.0) # phase gradient concentration cutoff
epochs = [(6,-1000,0),(8,-1000,0)]

# I want to use unicode greek leters instead of computer modern
# Need to switch to a font that can display them.

FONT = 'DejaVu Sans'
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif=FONT) 
matplotlib.rc('text', usetex='false') 
matplotlib.rcParams.update({'font.size': 12})


try:
    cache 
except:
    print "about to re-initialize cache, press enter to proceed"
    cache = {}
# todo should be disk cached as well?

close('all')
figure(figsize=( 9.8,5.9))
clf()
subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.1,hspace=0.6,wspace=.3)

for isa,(session,area) in en|todo:
    
    if (session,area) in cache:
        alldata,annotations = cache[session,area]
    else:
        annotations = []
        alldata = []
        for epoch in epochs:
            # get beta band
            epoch = (8,-1000,0)
            betapeak = get_stored_beta_peak(session,area,epoch)
            fa = betapeak-bandwidth*0.5, betapeak+bandwidth*o,5
            print 'beta peak at ',betapeak
            print 'fa,fb=',fa,fb
            sessiondata = []
            for s,a in sessions_areas():
                if not a==area: continue
                if not session[:3] in s: continue
                for trial in get_good_trials(s):
                    x = get_array_packed_lfp_analytic(s,a,trial,epoch,fa,fb)[...,::SKIP]
                    trialdata = x.transpose(2,0,1)
                    sessiondata.extend(trialdata)
                    annotations.append((s,a,epoch,trial))
                alldata.extend(sessiondata)
        alldata = ar|alldata
        cache[session,area] = (alldata,annotations)
    
    # get wave statistics
    wl    = array_wavelength_pgd_threshold(alldata.T,thresh=RCUT)
    amp   = mean(abs(alldata),axis=(1,2))
    sigma = array_phasegradient_magnitude_sigma(alldata.T)
    cv    = array_phasegradient_magnitude_cv(alldata.T)

    # We automatically search for wave examples
    # within certain ranges of amplitudes and wavelengths
    # the two monkeys have slightly different statistics, so
    # we tune this to match. Should really just pick data by hand
    # but there's too much to sort through
    if 'RUS' in session:
        abins = [ 0,25,33,45,60] 
        bins  = [ 0,14,17,23,30]
    else:
        abins = [0,10,20,30,100]
        bins  = [0,10,15,20,100]
    K = len(bins)-1

    subplot(2,K+1,1+5*isa)
    use = isfinite(wl)
    x = amp[use]
    y = wl[use]
    scatter(x,y,color='k',marker='o',s=1)
    
    print session, pearsonr(x,y)
    print session, description(y)
    
    threshold = 1.5*std(alldata)
    high = x>threshold
    print 'High beta',description(y[ high])
    print 'Low beta',description(y[~high])
    
    
    xticks(xlim())
    yticks(ylim())
    nicex()
    nicey()
    xlabel(u'Mean β amplitude\nμV',fontsize=10,fontname=FONT)
    ylabel('Wavelength (mm)',fontsize=10,fontname=FONT)
    fudgex(0)
    fudgey(3)
    simpleaxis()
    draw()
    title('Monkey %s area %s'%(session[0],area),fontsize=12,loc='left',y=1.08)

    for i,(a,b,c,d) in en|zip(bins[:-1],bins[1:],abins[:-1],abins[1:]):
        use2 = (wl>a)&(wl<b)&(amp>c)&(amp<d)
        print i, a, b, c, d, sum(use2)
        if not sum(use2): break
        
        keep = array(cv)
        keep[~use2] = NaN
        best = nanargmin(keep)
        print '>>',best, wl[best],amp[best]
        t = best
        
        subplot(2,K+1,i+2+5*isa)
        data = angle(alldata[t,...])
        data  = (data+4*pi-angle(mean(exp(1j*data))))%(2*pi)
        W,H = shape(data)
        Wmm,Hmm = W*0.4, H*0.4
        data = (data-pi/4)%(2*pi)
        imshow(data,vmin=0,vmax=2*pi,cmap=double_isolum,origin='upper',interpolation='nearest',extent=(0,Hmm,0,Wmm))
        for ii in linspace(0,10,11):
            axvline(ii*4.0/10,color='w',lw=0.3)
            axhline(ii*4.0/10,color='w',lw=0.3)
        xlim(0,Hmm)
        ylim(0,Wmm)
        nicex()
        nicey()
        xlabel('mm')
        ylabel('mm')
        fudgex()
        fudgey(3)
        cax = good_colorbar(0,2*pi,double_isolum,'$\\varphi$ (radians)',sideways=0)
        fudgey(4,cax)
    
        if i!=3:
            gcf().delaxes(cax)
        else:
            cx = gca()
            sca(cax)
            yticks([0,2*pi],['0',u'π'],fontname=FONT)
            sca(cx)
        if i!=0:
            ylabel('')
            yticks([])
            
        title(u'λ=%0.2g mm\nAmplitude=%0.2g μV'%(wl[t],amp[t]),fontname=FONT)

savefig('20160131_plane_wavelength_examples_with_statistics.pdf')




'''
stat results

(Pearson $\rho=$0.64 for monkey R, $\rho=$0.53 for monkey S)

3.2 mm to 28 mm, median 7.2 mm
median 11 mm for high beta
median 6.8 mm for low beta

3.5 mm to 31 mm, median 10 mm
median 13 for high beta
median 8.8 for low beta

RUS (0.63921326, 1.2324745150331624e-57)
RUS  std=3.93  q1=5.56  q3=7.20  skewness=1.46  p99=19.83  min=3.23  p90=13.96  max=28.45  p025=3.86  median=7.20  p95=17.02  N=490.00  p10=4.70  p01=3.59  q2=10.28  p975=17.87  p05=4.21  variance=15.51  kurtosis=2.52  mean=8.40 
High beta  std=4.72  q1=8.48  q3=11.47  skewness=0.75  p99=26.87  min=4.31  p90=17.55  max=28.45  p025=5.40  median=11.47  p95=19.69  N=100.00  p10=6.25  p01=4.95  q2=15.25  p975=21.67  p05=5.68  variance=22.49  kurtosis=0.75  mean=12.14 
Low beta  std=3.04  q1=5.27  q3=6.75  skewness=1.48  p99=17.62  min=3.23  p90=11.52  max=20.07  p025=3.73  median=6.75  p95=13.61  N=390.00  p10=4.49  p01=3.41  q2=8.69  p975=16.48  p05=4.11  variance=9.27  kurtosis=2.41  mean=7.44 

SPK (0.53209811, 6.8171137626194234e-55)
SPK  std=4.01  q1=7.77  q3=10.07  skewness=1.07  p99=24.04  min=3.50  p90=16.25  max=30.65  p025=5.29  median=10.07  p95=18.02  N=734.00  p10=6.37  p01=4.93  q2=12.98  p975=20.75  p05=5.78  variance=16.12  kurtosis=1.55  mean=10.81 
High beta  std=4.06  q1=10.48  q3=12.73  skewness=0.82  p99=25.08  min=5.09  p90=18.29  max=30.65  p025=6.98  median=12.73  p95=20.92  N=280.00  p10=8.18  p01=6.50  q2=15.67  p975=22.98  p05=7.76  variance=16.52  kurtosis=1.13  mean=13.29 
Low beta  std=3.12  q1=7.05  q3=8.81  skewness=1.36  p99=19.67  min=3.50  p90=13.36  max=25.30  p025=5.10  median=8.81  p95=14.94  N=454.00  p10=5.95  p01=4.63  q2=10.71  p975=16.44  p05=5.51  variance=9.75  kurtosis=3.25  mean=9.28 
'''


