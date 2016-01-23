
import cgid.config 
from numpy import *
import os

############################################################################
# Gather usable unit info 
# NOTE!! 1 INDEXED UNIT IDS !!
#
# defines a list of called units containing (session,area,uid) tuples
# execfile(expanduser('~/Workspace2/cgidanalysis_r2/grabunits.py'))
allunitsbysession = {
('SPK120918', 'PMv'): [165, 138, 155, 226, 48, 108, 118, 53, 101, 51, 37, 
    127, 116, 27, 54, 205, 227, 208, 160, 38, 218, 223, 222, 135, 206, 235,
    50, 168, 19, 184, 174, 198, 39, 4, 188, 185, 22, 187, 171, 114, 87, 150,
    84, 81, 12, 193, 199, 1, 236, 221, 157, 212, 63], 

('RUS120518', 'M1' ): [1, 35, 80, 30, 50, 55, 5, 78, 45, 121, 127, 18, 94,
    9, 122, 106, 118, 64, 65, 99, 102, 75, 107, 110, 68, 98, 88, 91, 73,
    66, 20, 42, 6], 

('SPK120925', 'PMv'): [8, 69, 86, 17, 62, 106, 195, 201, 143, 32, 211, 133,
    58, 212, 196, 76, 83, 210, 26, 103, 190, 139, 171, 163, 160, 122, 18,
    31, 157, 87, 220, 215, 181, 167, 172, 191, 89, 12, 9, 95, 80, 189, 140,
    25, 22, 54, 94, 178, 108, 1, 85, 41, 179, 159, 52, 35, 82, 113, 166,
    30, 4, 20], 

('RUS120523', 'M1' ): [80, 64, 98, 142, 59, 6, 63, 92, 101, 71, 76, 17,
    146, 74, 106, 105, 81, 119, 118, 43, 87, 128, 124, 129, 38, 133, 123,
    21, 114, 18, 5, 104, 1, 28, 130], 

('SPK120925', 'PMd'): [14, 123, 12, 67, 85, 16, 91, 8, 78, 43, 7, 41, 19,
    106, 109, 64, 11, 30, 104, 86, 34, 98, 57, 112, 70, 62, 113, 102, 103,
    20, 51, 3, 68, 22, 33, 45, 107, 115, 116, 73, 50, 46, 35], 

('SPK120924', 'PMv'): [110, 76, 149, 72, 12, 20, 5, 147, 14, 2, 87, 123, 65,
    31, 102, 193, 17, 114, 39, 42, 135, 108, 150, 10, 104, 51, 36, 55, 120,
    79, 134, 208, 61, 199, 140, 152, 74, 116, 46, 146, 169, 194, 6, 200,
    214, 155, 188, 50, 1, 90, 22, 162, 130, 26, 18, 175, 159, 221, 166, 184,
    91, 86, 177, 189, 165, 220, 185, 211, 141], 

('RUS120523', 'PMd'): [118, 36, 60, 75, 114, 4, 1, 121, 39, 23, 12, 108, 11,
    117, 59, 22, 119, 81, 71, 37, 99, 56, 34, 104, 125, 30, 54, 92, 112, 65,
    61, 62, 33, 97, 78, 96, 55], 

('SPK120924', 'M1' ): [93, 40, 26, 110, 35, 80, 11, 134, 54, 43, 41, 4, 62,
    100, 90, 6, 107, 112, 18, 92, 16, 83, 49, 30, 82, 61, 87, 56, 25, 53,
    47, 36, 104], 

('SPK120925', 'M1' ): [8, 30, 86, 105, 111, 37, 107, 10, 21, 27, 22, 33,
    51, 95, 85, 59, 46, 52, 19, 98, 63, 13, 15, 90, 56, 81, 109, 80, 77,
    73, 48, 17, 14, 5, 74, 58, 84], 

('RUS120518', 'PMv'): [46, 25, 163, 210, 200, 180, 172, 169, 141, 22, 184,
    100, 36, 90, 34, 18, 161, 145, 21, 136, 6, 2, 111, 29, 49, 165, 62, 50,
    38, 162, 77, 16, 115, 1, 101, 94, 15], 

('RUS120521', 'PMd'): [45, 69, 77, 85, 25, 67, 89, 98, 94, 65, 27, 59, 51,
    26, 42, 30, 19, 99, 31, 13, 62, 93, 74], 

('SPK120924', 'PMd'): [25, 115, 23, 92, 20, 99, 100, 90, 7, 81, 8, 111,
    117, 35, 91, 112, 65, 48, 59, 16, 104, 105, 38, 63, 83, 122, 108, 33,
    107, 75, 72, 53, 3, 11, 69, 114, 19, 46, 45, 97, 102, 120, 80, 41, 42,
    87, 96, 124], 

('SPK120918', 'PMd'): [119, 70, 45, 42, 40, 84, 43, 36, 62, 12, 14, 35, 77,
    49, 4, 126, 74, 98, 109, 69, 50, 22, 122, 54, 115, 82, 88, 55, 32, 97,
    23, 83, 111, 19, 61, 72, 104, 17, 94, 108, 87, 131, 121], 

('RUS120523', 'PMv'): [26, 103, 109, 198, 191, 190, 136, 181, 139, 184,
    202, 199, 65, 36, 32, 174, 149, 92, 59, 23, 172, 41, 80, 165, 5, 164,
    104, 157, 77, 53, 49, 47, 28, 46, 99, 44, 58, 37, 118, 123, 14, 97,
    148, 17, 86, 22, 94], 

('RUS120521', 'PMv'): [38, 43, 196, 191, 155, 166, 185, 4, 186, 147, 161,
    136, 154, 105, 75, 10, 23, 104, 163, 24, 87, 133, 19, 160, 34, 29, 57,
    2, 158, 47, 56, 15, 110, 81, 45, 41, 35, 156], 

('RUS120521', 'M1' ): [105, 57, 35, 55, 51, 42, 100, 64, 93, 62, 102, 86,
    37, 101, 24, 11, 31, 96, 91, 73, 66, 97, 26, 3, 81, 19, 118, 111], 

('SPK120918', 'M1' ): [6, 107, 82, 80, 90, 83, 23, 39, 52, 2, 65, 46, 121,
    79, 96, 102, 118, 21, 49, 86, 93, 73, 19, 55, 59, 70, 33], 

('RUS120518', 'PMd'): [95, 98, 8, 68, 110, 94, 44, 105, 86, 30, 69, 75, 21,
    103, 83, 72, 22, 38, 93, 34, 36, 96, 77, 59, 79, 73, 90, 113, 82, 91, 89]
}
units = []
for (session,area), uu in allunitsbysession.iteritems():
    units+=[(session,area,u) for u in uu]
import pickle
classification_results = pickle.load(open(cgid.config.CGID_PACKAGE + os.path.sep + 'classification_results.p','rb'))
colorcodes = array([[ 0. , 0.6, 1.0],
                    [ 0.9, 0. , 1.0]])

print 'LOADED CLASSIFICATION RESULTS: CAUTION ZERO INDEXED (UNUSUAL, NOT LIKE MATLAB)'




