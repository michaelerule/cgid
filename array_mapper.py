#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division


import os

from numpy                    import *
from numpy.linalg.linalg      import inv
from matplotlib.mlab          import find

from neurotools.tools         import memoize
from neurotools.spatial.array import trim_array,trim_array_as_if

from cgid.config              import CGID_PACKAGE
from cgid.data_loader         import get_array_map

def getElectrodePositions(session,area):
    # now comes the hard part: identifying electrode locations in 
    # cortical space. We use the same mapping function as the video
    # this generates a basis, we still need to convert the array into 
    # that basis
    import pickle
    corners = pickle.load(open(CGID_PACKAGE+os.path.sep+'new_corners.p','rb'))
    
    # note: run compile_array_maps.py to build this
    maps    = pickle.load(open(CGID_PACKAGE+os.path.sep+'maps.p','rb'))
    monkey  = session[:3]
    arrayChannelMap = maps[monkey][area]

    availableChannels = int32(zeros(96))
    foundchannels = set(ravel(arrayChannelMap))
    if -1 in foundchannels:
        foundchannels.remove(-1)
    availableChannels[array(list(foundchannels))-1]=1

    def getMatrixFromAnatomicalToImage(monkey):
        scale = 3.6 # size of M1 array in MM 
        if monkey=='RUS':
            A,D,C,B = corners[monkey]['M1']
            origin = B
            y_vector = C-origin
            x_vector = A-B
            basis = array([x_vector,y_vector])
        if monkey=='SPK':
            A,D,C,B  = corners[monkey]['M1']   
            origin   = D
            y_vector = C-origin
            x_vector = A-D
            basis = array([x_vector,y_vector])
        basis /= scale
        return origin,basis

    # This quadrilateral defines the physical locations of four points of 
    # the array, starting at the top left, and proceeding counter-clockwise
    # around the array as specified in arrayChannelMap
    quad = array(corners[monkey][area])
    origin,basis = getMatrixFromAnatomicalToImage(monkey)
    anatomical = inv(basis)
    quad = (quad-origin)
    quad = (quad.dot(anatomical))

    # Need to interpolate in the quadralateral to get electrode positions
    # consider making this a subroutine
    positions = {}
    nrows,ncols = shape(arrayChannelMap)
    topleft,bottomleft,bottomright,topright = quad
    for chi in find(availableChannels)+1:
        row = find([any(r==chi) for r in arrayChannelMap])
        col = find([any(c==chi) for c in arrayChannelMap.T])
        if prod(shape(row))<1 or prod(shape(col))<1:
            print 'error cannot locate channel %d'%chi
            continue
        # The *2+1 accounts for the fact that the electrodes are in 
        # the center of the square patches -- our quadrilateral defines the
        # outer boundary of the array, not the electrodes
        row_fraction = (row*2+1)/float(nrows*2)
        col_fraction = (col*2+1)/float(ncols*2)
        row_b = row_fraction*(bottomleft-topleft)
        col_b = col_fraction*(topright  -topleft)
        position = row_b+col_b+topleft
        positions[chi] = position
    
    return quad,positions

def get_electrode_positions(session, area, outline=None, 
    electrode_spacing=0.4):
    '''
    Retrieves positions of electrodes in the array in anatomical coordinates.
    
    If an array has empty rows or columns at the edge, these rows and 
    columnds are trimmed when packing data into the array. This code also
    trims array maps to account for this, and will return an array of 
    position data that can be indexed in the same manner as the array-packed
    data.
    
    For an example 10x10 array, this will return an array of the positiond
    marked with 'o' below.
    
    Note: Array wire bundle exits to the right
    
     0mm                                     4mm
     0   1   2   3   4   5   6   7   8   9  10
     ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
     +---+---+---+---+---+---+---+---+---+---+ 0
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---+---+---+---+---+---+---+---+---+---+
     0                                         1
    '''
    if outline is None: outline,_=getElectrodePositions(session,area)
    topleft, bottomleft, bottomright, topright = outline
    dx = topright  -topleft
    dy = bottomleft-topleft
    
    M = get_array_map(session,area)
    nrow,ncol = shape(M)
    
    positions = array([[
    
        row + col + topleft
        for row in (arange(0,nrow)+0.5)[:,None]*dx[None,:]/nrow]
        for col in (arange(0,ncol)+0.5)[:,None]*dy[None,:]/ncol])
    
    positions = trim_array_as_if(M,positions)
    dx /= ncol
    dy /= nrow
    return positions,dx,dy


def get_interelectrode_positions(session, area, outline=None,
    electrode_spacing=0.4):
    '''
    Retrieve information needed to plot array phase gradient derivatives 
    in anatomical coordinates.
    
    Array phase gradients are computed in the array coordinates. We need 
    the positions of the points between groups of 4 electrodes, as well
    as X and Y units vectors, to plot them correctly in anatomical 
    coordinates.
    
    If an array has empty rows or columns at the edge, these rows and 
    columnds are trimmed when packing data into the array. This code also
    trims array maps to account for this, and will return an array of 
    position data that can be indexed in the same manner as the array-packed
    data.
    
    Uses the stored array map outline to get anatomical coordinates
    of the positions between electrodes.
    Returns a dictionary, with 0-index row,column position as the keys
    and anatomical coordiantes as the values.
    Also returns a pair of transformed unit vectors, so that phase gradient
    vector fields can be plotted with the correct orientation.
    
     Arays are 4x4 mm but the electrodes are at the center, so the electrode
     coverage is really only about 3.6x3.6mm. The derivaties are estimated
     at neighborhoods of 4 electrodes. 
    
     So I guess they are located at arange(1,10)/10.
     
     Note: Array wire bundle exits to the right
    
     0mm                                     4mm
     0   1   2   3   4   5   6   7   8   9  10
     ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
     +---+---+---+---+---+---+---+---+---+---+ 0
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---x---x---x---x---x---x---x---x---x---+
     | o | o | o | o | o | o | o | o | o | o |
     +---+---+---+---+---+---+---+---+---+---+
     0                                         1
    '''
    positions,dx,dy = get_electrode_positions(session, area, outline)
    positions = 0.25*(
        + positions[1: ,1: ]
        + positions[:-1,1: ]
        + positions[1: ,:-1]
        + positions[:-1,:-1])
    return positions,dx,dy














