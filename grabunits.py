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
from collections import defaultdict

print('identifying units')
print('WARNING: REMEMBER UNITS ARE 1 INDEXED')
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
        print('no')
    narea = areas.index(area)+1
    units.append((session,area,uid))

allunitsbysession = defaultdict(list)

for session,area,uid in units:
    allunitsbysession[session,area].append(uid)

acceptable = {('RUS120518', 'M1', 1),
 ('RUS120518', 'M1', 65),
 ('RUS120518', 'M1', 66),
 ('RUS120518', 'M1', 68),
 ('RUS120518', 'M1', 75),
 ('RUS120518', 'M1', 94),
 ('RUS120518', 'M1', 106),
 ('RUS120518', 'M1', 107),
 ('RUS120518', 'M1', 110),
 ('RUS120518', 'PMd', 21),
 ('RUS120518', 'PMd', 22),
 ('RUS120518', 'PMd', 30),
 ('RUS120518', 'PMd', 36),
 ('RUS120518', 'PMd', 59),
 ('RUS120518', 'PMd', 72),
 ('RUS120518', 'PMd', 73),
 ('RUS120518', 'PMd', 75),
 ('RUS120518', 'PMd', 77),
 ('RUS120518', 'PMd', 89),
 ('RUS120518', 'PMd', 90),
 ('RUS120518', 'PMd', 91),
 ('RUS120518', 'PMd', 95),
 ('RUS120518', 'PMv', 15),
 ('RUS120518', 'PMv', 16),
 ('RUS120518', 'PMv', 38),
 ('RUS120518', 'PMv', 46),
 ('RUS120518', 'PMv', 49),
 ('RUS120518', 'PMv', 50),
 ('RUS120518', 'PMv', 90),
 ('RUS120518', 'PMv', 101),
 ('RUS120518', 'PMv', 141),
 ('RUS120521', 'M1', 24),
 ('RUS120521', 'M1', 26),
 ('RUS120521', 'M1', 31),
 ('RUS120521', 'M1', 37),
 ('RUS120521', 'M1', 62),
 ('RUS120521', 'M1', 64),
 ('RUS120521', 'M1', 66),
 ('RUS120521', 'M1', 86),
 ('RUS120521', 'M1', 91),
 ('RUS120521', 'M1', 93),
 ('RUS120521', 'M1', 96),
 ('RUS120521', 'M1', 97),
 ('RUS120521', 'M1', 101),
 ('RUS120521', 'M1', 102),
 ('RUS120521', 'PMd', 13),
 ('RUS120521', 'PMd', 26),
 ('RUS120521', 'PMd', 27),
 ('RUS120521', 'PMd', 31),
 ('RUS120521', 'PMd', 42),
 ('RUS120521', 'PMd', 45),
 ('RUS120521', 'PMd', 51),
 ('RUS120521', 'PMd', 59),
 ('RUS120521', 'PMd', 62),
 ('RUS120521', 'PMd', 67),
 ('RUS120521', 'PMd', 85),
 ('RUS120521', 'PMd', 98),
 ('RUS120521', 'PMd', 99),
 ('RUS120521', 'PMv', 4),
 ('RUS120521', 'PMv', 23),
 ('RUS120521', 'PMv', 34),
 ('RUS120521', 'PMv', 35),
 ('RUS120521', 'PMv', 56),
 ('RUS120521', 'PMv', 57),
 ('RUS120521', 'PMv', 81),
 ('RUS120521', 'PMv', 87),
 ('RUS120521', 'PMv', 185),
 ('RUS120523', 'M1', 1),
 ('RUS120523', 'M1', 28),
 ('RUS120523', 'M1', 38),
 ('RUS120523', 'M1', 43),
 ('RUS120523', 'M1', 71),
 ('RUS120523', 'M1', 74),
 ('RUS120523', 'M1', 76),
 ('RUS120523', 'M1', 81),
 ('RUS120523', 'M1', 92),
 ('RUS120523', 'M1', 106),
 ('RUS120523', 'M1', 114),
 ('RUS120523', 'M1', 118),
 ('RUS120523', 'M1', 119),
 ('RUS120523', 'M1', 123),
 ('RUS120523', 'M1', 129),
 ('RUS120523', 'M1', 130),
 ('RUS120523', 'M1', 133),
 ('RUS120523', 'PMd', 11),
 ('RUS120523', 'PMd', 12),
 ('RUS120523', 'PMd', 23),
 ('RUS120523', 'PMd', 34),
 ('RUS120523', 'PMd', 36),
 ('RUS120523', 'PMd', 39),
 ('RUS120523', 'PMd', 54),
 ('RUS120523', 'PMd', 55),
 ('RUS120523', 'PMd', 56),
 ('RUS120523', 'PMd', 60),
 ('RUS120523', 'PMd', 62),
 ('RUS120523', 'PMd', 65),
 ('RUS120523', 'PMd', 71),
 ('RUS120523', 'PMd', 96),
 ('RUS120523', 'PMd', 97),
 ('RUS120523', 'PMd', 114),
 ('RUS120523', 'PMd', 117),
 ('RUS120523', 'PMd', 118),
 ('RUS120523', 'PMd', 119),
 ('RUS120523', 'PMv', 5),
 ('RUS120523', 'PMv', 22),
 ('RUS120523', 'PMv', 36),
 ('RUS120523', 'PMv', 46),
 ('RUS120523', 'PMv', 47),
 ('RUS120523', 'PMv', 58),
 ('RUS120523', 'PMv', 59),
 ('RUS120523', 'PMv', 86),
 ('RUS120523', 'PMv', 92),
 ('RUS120523', 'PMv', 94),
 ('RUS120523', 'PMv', 118),
 ('RUS120523', 'PMv', 123),
 ('RUS120523', 'PMv', 139),
 ('RUS120523', 'PMv', 148),
 ('RUS120523', 'PMv', 149),
 ('RUS120523', 'PMv', 157),
 ('RUS120523', 'PMv', 199),
 ('SPK120918', 'M1', 2),
 ('SPK120918', 'M1', 23),
 ('SPK120918', 'M1', 80),
 ('SPK120918', 'PMd', 12),
 ('SPK120918', 'PMd', 23),
 ('SPK120918', 'PMd', 35),
 ('SPK120918', 'PMd', 36),
 ('SPK120918', 'PMd', 43),
 ('SPK120918', 'PMd', 50),
 ('SPK120918', 'PMd', 55),
 ('SPK120918', 'PMd', 62),
 ('SPK120918', 'PMd', 70),
 ('SPK120918', 'PMd', 72),
 ('SPK120918', 'PMd', 83),
 ('SPK120918', 'PMd', 84),
 ('SPK120918', 'PMd', 88),
 ('SPK120918', 'PMd', 108),
 ('SPK120918', 'PMd', 109),
 ('SPK120918', 'PMd', 111),
 ('SPK120918', 'PMd', 119),
 ('SPK120918', 'PMd', 122),
 ('SPK120918', 'PMv', 38),
 ('SPK120918', 'PMv', 51),
 ('SPK120918', 'PMv', 53),
 ('SPK120918', 'PMv', 84),
 ('SPK120918', 'PMv', 116),
 ('SPK120918', 'PMv', 165),
 ('SPK120918', 'PMv', 185),
 ('SPK120918', 'PMv', 188),
 ('SPK120918', 'PMv', 199),
 ('SPK120918', 'PMv', 205),
 ('SPK120918', 'PMv', 208),
 ('SPK120918', 'PMv', 221),
 ('SPK120918', 'PMv', 222),
 ('SPK120918', 'PMv', 223),
 ('SPK120918', 'PMv', 236),
 ('SPK120924', 'M1', 4),
 ('SPK120924', 'M1', 16),
 ('SPK120924', 'M1', 25),
 ('SPK120924', 'M1', 26),
 ('SPK120924', 'M1', 35),
 ('SPK120924', 'M1', 40),
 ('SPK120924', 'M1', 43),
 ('SPK120924', 'M1', 49),
 ('SPK120924', 'M1', 53),
 ('SPK120924', 'M1', 54),
 ('SPK120924', 'M1', 56),
 ('SPK120924', 'M1', 61),
 ('SPK120924', 'M1', 62),
 ('SPK120924', 'M1', 80),
 ('SPK120924', 'M1', 83),
 ('SPK120924', 'M1', 93),
 ('SPK120924', 'M1', 100),
 ('SPK120924', 'M1', 104),
 ('SPK120924', 'M1', 110),
 ('SPK120924', 'M1', 112),
 ('SPK120924', 'PMd', 3),
 ('SPK120924', 'PMd', 7),
 ('SPK120924', 'PMd', 8),
 ('SPK120924', 'PMd', 16),
 ('SPK120924', 'PMd', 20),
 ('SPK120924', 'PMd', 23),
 ('SPK120924', 'PMd', 25),
 ('SPK120924', 'PMd', 33),
 ('SPK120924', 'PMd', 35),
 ('SPK120924', 'PMd', 41),
 ('SPK120924', 'PMd', 42),
 ('SPK120924', 'PMd', 45),
 ('SPK120924', 'PMd', 48),
 ('SPK120924', 'PMd', 53),
 ('SPK120924', 'PMd', 59),
 ('SPK120924', 'PMd', 69),
 ('SPK120924', 'PMd', 72),
 ('SPK120924', 'PMd', 75),
 ('SPK120924', 'PMd', 81),
 ('SPK120924', 'PMd', 90),
 ('SPK120924', 'PMd', 92),
 ('SPK120924', 'PMd', 97),
 ('SPK120924', 'PMd', 104),
 ('SPK120924', 'PMd', 105),
 ('SPK120924', 'PMd', 112),
 ('SPK120924', 'PMd', 114),
 ('SPK120924', 'PMd', 115),
 ('SPK120924', 'PMd', 124),
 ('SPK120924', 'PMv', 1),
 ('SPK120924', 'PMv', 31),
 ('SPK120924', 'PMv', 42),
 ('SPK120924', 'PMv', 50),
 ('SPK120924', 'PMv', 51),
 ('SPK120924', 'PMv', 74),
 ('SPK120924', 'PMv', 87),
 ('SPK120924', 'PMv', 91),
 ('SPK120924', 'PMv', 102),
 ('SPK120924', 'PMv', 110),
 ('SPK120924', 'PMv', 114),
 ('SPK120924', 'PMv', 116),
 ('SPK120924', 'PMv', 120),
 ('SPK120924', 'PMv', 140),
 ('SPK120924', 'PMv', 141),
 ('SPK120924', 'PMv', 147),
 ('SPK120924', 'PMv', 159),
 ('SPK120924', 'PMv', 162),
 ('SPK120924', 'PMv', 166),
 ('SPK120924', 'PMv', 185),
 ('SPK120924', 'PMv', 188),
 ('SPK120924', 'PMv', 193),
 ('SPK120924', 'PMv', 220),
 ('SPK120924', 'PMv', 221),
 ('SPK120925', 'M1', 10),
 ('SPK120925', 'M1', 13),
 ('SPK120925', 'M1', 14),
 ('SPK120925', 'M1', 15),
 ('SPK120925', 'M1', 19),
 ('SPK120925', 'M1', 21),
 ('SPK120925', 'M1', 22),
 ('SPK120925', 'M1', 30),
 ('SPK120925', 'M1', 37),
 ('SPK120925', 'M1', 48),
 ('SPK120925', 'M1', 51),
 ('SPK120925', 'M1', 56),
 ('SPK120925', 'M1', 58),
 ('SPK120925', 'M1', 59),
 ('SPK120925', 'M1', 73),
 ('SPK120925', 'M1', 74),
 ('SPK120925', 'M1', 77),
 ('SPK120925', 'M1', 84),
 ('SPK120925', 'M1', 85),
 ('SPK120925', 'M1', 86),
 ('SPK120925', 'M1', 98),
 ('SPK120925', 'M1', 105),
 ('SPK120925', 'PMd', 3),
 ('SPK120925', 'PMd', 7),
 ('SPK120925', 'PMd', 8),
 ('SPK120925', 'PMd', 11),
 ('SPK120925', 'PMd', 14),
 ('SPK120925', 'PMd', 20),
 ('SPK120925', 'PMd', 22),
 ('SPK120925', 'PMd', 30),
 ('SPK120925', 'PMd', 33),
 ('SPK120925', 'PMd', 34),
 ('SPK120925', 'PMd', 35),
 ('SPK120925', 'PMd', 41),
 ('SPK120925', 'PMd', 43),
 ('SPK120925', 'PMd', 45),
 ('SPK120925', 'PMd', 46),
 ('SPK120925', 'PMd', 50),
 ('SPK120925', 'PMd', 51),
 ('SPK120925', 'PMd', 57),
 ('SPK120925', 'PMd', 67),
 ('SPK120925', 'PMd', 68),
 ('SPK120925', 'PMd', 86),
 ('SPK120925', 'PMd', 98),
 ('SPK120925', 'PMd', 103),
 ('SPK120925', 'PMd', 104),
 ('SPK120925', 'PMd', 106),
 ('SPK120925', 'PMd', 107),
 ('SPK120925', 'PMd', 112),
 ('SPK120925', 'PMd', 113),
 ('SPK120925', 'PMd', 116),
 ('SPK120925', 'PMd', 123),
 ('SPK120925', 'PMv', 25),
 ('SPK120925', 'PMv', 26),
 ('SPK120925', 'PMv', 30),
 ('SPK120925', 'PMv', 35),
 ('SPK120925', 'PMv', 52),
 ('SPK120925', 'PMv', 54),
 ('SPK120925', 'PMv', 82),
 ('SPK120925', 'PMv', 83),
 ('SPK120925', 'PMv', 85),
 ('SPK120925', 'PMv', 86),
 ('SPK120925', 'PMv', 94),
 ('SPK120925', 'PMv', 103),
 ('SPK120925', 'PMv', 108),
 ('SPK120925', 'PMv', 133),
 ('SPK120925', 'PMv', 160),
 ('SPK120925', 'PMv', 163),
 ('SPK120925', 'PMv', 166),
 ('SPK120925', 'PMv', 167),
 ('SPK120925', 'PMv', 171),
 ('SPK120925', 'PMv', 172),
 ('SPK120925', 'PMv', 178),
 ('SPK120925', 'PMv', 179),
 ('SPK120925', 'PMv', 189),
 ('SPK120925', 'PMv', 190),
 ('SPK120925', 'PMv', 191),
 ('SPK120925', 'PMv', 195),
 ('SPK120925', 'PMv', 210),
 ('SPK120925', 'PMv', 212),
 ('SPK120925', 'PMv', 220)}

