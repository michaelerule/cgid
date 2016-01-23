
# this ... might be able to install missing dependencies if we're lucky
from cgid.dependencies import *


import sys
import os
from   os.path import *
from   collections import *
from   itertools   import *
from   multiprocessing import Process, Pipe, cpu_count, Pool
import pickle
from   pylab       import *
import random

from   sklearn.metrics import roc_auc_score,roc_curve,auc

import scipy
from   scipy.io     import *
from   scipy.interpolate import *
import scipy.optimize
from   scipy.optimize import leastsq
from   scipy.signal import *
from   scipy.stats  import wilcoxon

set_printoptions(precision=2)

############################################################################
############################################################################

from cgid.config import *

# the following imports are essential
from cgid.visualize import *
from cgid.data_loader import *
from cgid.tools import *
from cgid.spikes import *
from cgid.lfp import *

# the following imports may be optional
from cgid.phasetools import *
from cgid.phaseplane import *
from cgid.array_mapper import *
from cgid.beta_events import *
from cgid.circulardistributions import *
from cgid.waveparametrics import *


# This was previously needed to load in the unit sorts / classes from elsewhere
# but that information should have been tranferred to the unitinfo.py script
# this may still be used to regenerate the unit classifications, but the
# path to my manual sorts will need to be set correctly. 
# from cgid.grabunits import *

from cgid.phase_plots import *
from cgid.unitinfo import *

import cgid.beta

############################################################################
# Gather usable unit info 
# DONT RUN THIS 

# units = get_cgid_unit_info()

# load results from disk
allresults = load_ppc_results_archives()


print 'warning, defining PPCFREQS globally'
PPCFREQS = abs(fftfreq(200,1./1000)[:101])

############################################################################
############################################################################

import neurotools
from neurotools.nlab import *




