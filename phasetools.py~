'''
Library of array statistics routines.
'''

from neurotools.tools import *

ELECTRODE_SPACING = 0.4

def population_kuromoto(population):
    warn('statistics computed over first axis. not for 2d array data')
    return sum(population/abs(population),axis=0)

def population_synchrony(population):
    warn('statistics computed over first axis. not for 2d array data')
    return abs(mean(population,axis=0))/mean(abs(population),axis=0)

def population_average_amplitude(population):
    warn('statistics computed over first axis. not for 2d array data')
    return mean(abs(frame),axis=0)

def array_average_ampltiude(frame):
    warn('using this with interpolated channels will bias amplitude')
    warn('this assumes first 2 dimensions are array axis')
    return mean(abs(frame),axis=(0,1))

def array_kuromoto(population):
    warn('using this with interpolated channels will inflate synchrony')
    warn('this assumes first 2 dimensions are array axis')
    return sum(population/abs(population),axis=(0,1))

def array_synchrony(population):
    warn('using this with interpolated channels will inflate synchrony')
    warn('this assumes first 2 dimensions are array axis')
    return abs(mean(population,axis=(0,1)))/mean(abs(population),axis=(0,1))

def phase_gradient_array(frame):
    dx = frame[1:,:-1,...]-frame[:-1,:-1,...]+pi
    dy = frame[:-1,1:,...]-frame[:-1,:-1,...]+pi
    return dx%(2*pi)+1j*(dy%(2*pi))-(1+1j)*pi

def array_pgd_upper(frame):
    '''
    The average gradient magnitude can be inflated if there is noise
    but provides an upper bound on spatial frequency ( lower bound on 
    wavelength ). 
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = phase_gradient_array(frame)
    return mean(abs(pg),axis=-1)

def array_pgd_lower(frame):
    '''
    The magnitude of the average gradient provides a very accurate estimate
    of wavelength even in the presence of noise. However, it will 
    understimate the phase gradient if the wave structure is not perfectly
    planar
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = phase_gradient_array(frame)
    return abs(mean(pg,axis=-1))

def array_wavelength_lower(frame):
    '''
    phase gradients are in units of radians per electrode
    we would like units of mm per cycle
    there are 2pi radians per cycle
    there are 0.4mm per electrode
    phase gradient / 2 pi is in units of cycles per electrode
    electrode spacing / (phase gradient / 2 pi)
    '''
    warn('this code will break if ELECTRODE_SPACING changes or is inconsistant across datasets')
    warn('using something other than mean may make this less sensitive to outliers and noise')
    return ELECTRODE_SPACING*2*pi/array_pgd_upper(frame)

def array_wavelength_upper(frame):
    '''
    phase gradients are in units of radians per electrode
    we would like units of mm per cycle
    there are 2pi radians per cycle
    there are 0.4mm per electrode
    phase gradient / 2 pi is in units of cycles per electrode
    electrode spacing / (phase gradient / 2 pi)
    '''
    warn('this code will break if ELECTRODE_SPACING changes or is inconsistant across datasets')
    warn('using something other than mean may make this less sensitive to outliers and noise')
    return ELECTRODE_SPACING*2*pi/array_pgd_lower(frame)

def array_synchrony_pgd(frame):
    '''
    The phase gradient directionality measure from Rubinto et al 2009 is
    abs(mean(pg))/mean(abs(pg))
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = phase_gradient_array(frame)
    return synchrony(pg)

def array_kuromoto_pgd(frame):
    '''
    A related directionality index ignores vector amplitude. Nice if 
    there are large outliers.
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = phase_gradient_array(frame)
    return kuromoto(pg)



    


