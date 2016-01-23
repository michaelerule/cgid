############################################################################
# Gather usable unit info 
# 1 INDEXED UNIT IDS
#
# This file is obsolete

from cgid.config import sortedunits
import pylab 
import scipy.io
import collections
import itertools
import sys
import os

print 'identifying units'
print 'WARNING: REMEMBER UNITS ARE 1 INDEXED'
allfiles=[]
allannotated=[]

for parent,dirs,files in os.walk(sortedunits):
    if parent==sortedunits: continue
    if 'low_snr' in parent: continue
    allfiles.extend(files)
    allannotated.extend([(parent.split('/')[-1],)+tuple(f.split('_')[:3]) for f in files])
units = []

for file in allfiles:
    session,area,uid,_,_ = file.split('_')[:5]
    uid = int(uid.split('(')[0])
    if 'SPK' in session:
        nmonkey  = 1
        nsession = 1+spike_sessions.index(session)
    elif 'RUS' in session:
        nmonkey  = 2
        nsession = 1+rusty_sessions.index(session)
    else:
        print 'no'
    narea = areas.index(area)+1
    units.append((session,area,uid))

allunitsbysession = defaultdict(list)

for session,area,uid in units:
    allunitsbysession[session,area].append(uid)
    



