from os.path import expanduser
import cgid

# These need to be reconfigured per system. Only two users right now
# so just hard code it 
import os
myhost = os.uname()[1]
if myhost=='moonbase':
    archive = expanduser('~/Workspace2/CGID_essential/')
else:
    print 'HI WILSON PLEASE ENTER THE PATH TO THE CGID ARCHIVES BELOW'
    archive = 'PATH/TO/WILSONS/COPY/OF/THE/CGID/ARCHIVES'

"""
This directory should contain the following files:
 'RUS120521_PMd.mat',
 'RUS120521_PMv.mat',
 'RUS120523_M1.mat',
 'SPK120924_M1.mat',
 'RUS120523_PMd.mat',
 'SPK120925_PMv.mat',
 'RUS120518_M1.mat',
 'SPK120924_PMd.mat',
 'SPK120918_M1.mat',
 'SPK120918_PMv.mat',
 'RUS120523_PMv.mat',
 'RUS120521_M1.mat',
 'RUS120518_PMd.mat',
 'RUS120518_PMv.mat',
 'SPK120925_M1.mat',
 'SPK120924_PMv.mat',
 'SPK120918_PMd.mat',
"""

# This needs to be defined if we want to load the unit sorts from my external copy on disk
# The current version as of 20160122 has been hard coded into the unitinfo module
# so this should not be needed anymore.
sortedunits    = expanduser('~/Workspace2/CGID_unit_classification/all_summary_by_epoch_waveforms_fixed_autocorrelation_ok_manually_sorted/')

# Formerly used path to datasets. Does not appear to be required anymore
# Preserving in case surprising bugs happen
#datadir        = '/ldisk_2/mrule/archive/'

# Some data is stored with the source code, so we need to know where 
# that's located
CGID_PACKAGE = os.path.dirname(cgid.__file__)

CGID_ARCHIVE   = archive
areas          = 'M1 PMv PMd'.split()
monkeys        = 'RUS SPK'.split()
sessions       = [['120518','120521','120523'],
                  ['120918','120924','120925']]
spike_sessions = 'SPK120918 SPK120924 SPK120925'.split()
rusty_sessions = 'RUS120518 RUS120521 RUS120523'.split()
sessionnames   = [rusty_sessions,spike_sessions]












