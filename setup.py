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

import sys
IN_SPHINX=False
if 'sphinx' in sys.modules:
    print('Inside Sphinx autodoc; NOT loading scipy and pylab namespaces!')
    IN_SPHINX=True    

# this ... might be able to install missing dependencies
# In the future we'll make this a package so the dependenies are
# handeled at installation
if not IN_SPHINX: from cgid.dependencies import *

import sys
import os
from   multiprocessing import Process, Pipe, cpu_count, Pool
import pickle
import random
from   sklearn.metrics import roc_auc_score,roc_curve,auc
import scipy
import scipy.optimize
from   scipy.optimize import leastsq
import numpy as np

if not IN_SPHINX: 
    from   os.path import *
    from   collections import *
    from   itertools   import *
    from   pylab       import *
    from   scipy.io     import *
    from   scipy.interpolate import *
    from   scipy.signal import *
    from   scipy.stats import *
    from   scipy.stats.stats import *

np.core.arrayprint.set_printoptions(precision=2)

################################################################
################################################################

from cgid.config import *
    
if not IN_SPHINX: 

    # the following imports are essential
    from cgid.visualize     import *
    from cgid.data_loader   import *
    from cgid.tools         import *
    from cgid.spikes        import *
    from cgid.lfp           import *

    # the following imports may be optional
    from cgid.phasetools                import *
    from cgid.phaseplane                import *
    from cgid.array_mapper              import *
    from cgid.array                     import *
    from cgid.waveparametrics           import *
    from cgid.plotting_helper_functions import *
    from cgid.plotting_unit_summary     import *
    from cgid.phase_plots               import *
    from cgid.unitinfo                  import *
    from cgid.beta                      import *

################################################################
# Gather usable unit info
# DONT RUN THIS

# units = get_cgid_unit_info()
if not IN_SPHINX: 
    # load results from disk
    allresults = load_ppc_results_archives()

    print('warning, defining PPCFREQS globally')
    try:
        PPCFREQS = abs(np.fft.helper.fftfreq(200,1./1000)[:101])
    except:
        PPCFREQS = None
        # sphinx bug workaround

################################################################
################################################################

import neurotools
import matplotlib
if not IN_SPHINX: 
    from neurotools.nlab import *
    from neurotools.color import *

try:
    matplotlib.rcParams['axes.titlesize']=11
    matplotlib.rcParams['axes.labelsize']=9
    matplotlib.rcParams['axes.facecolor']=(1,)*4
    matplotlib.rcParams['axes.linewidth']=0.8
    matplotlib.rcParams['figure.figsize']=(7.5,7.0)
    matplotlib.rcParams['figure.facecolor']=(1,)*4
    matplotlib.rcParams['lines.solid_capstyle']='projecting'
    matplotlib.rcParams['lines.solid_joinstyle']='miter'
    matplotlib.rcParams['xtick.labelsize']=9
    matplotlib.rcParams['ytick.labelsize']=9
    matplotlib.rcParams['figure.subplot.bottom']=0.08
    matplotlib.rcParams['figure.subplot.hspace']=0.45
    matplotlib.rcParams['figure.subplot.left'  ]=0.08
    matplotlib.rcParams['figure.subplot.right' ]=0.98
    matplotlib.rcParams['figure.subplot.top'   ]=0.9
    matplotlib.rcParams['figure.subplot.wspace']=0.35
except:
    print('Static configuration of matplotlib failed')
    print('Are we inside Sphinx autodoc?')
