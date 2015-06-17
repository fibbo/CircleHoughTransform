'''
Created on Apr 29, 2015

@author: phi
'''
from Tools import readFile2, S_ERROR, S_OK
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import itertools
import numpy as np
from timer import Timer

NUMBER_OF_R_BINS = 200 #bins for radius
NUMBER_OF_S_BINS = 1000 #bins for space


def SHistogram(data, number_of_bins, hrange=None):
  """ Creates a histogram for a given with number_of_bins entries within the boundaries given by range. 
      The important thing is that in addition to the histogram we also create an array that stores
      the single data objects that were filled in the histogram in the same bin they have been place
      in the histogram. So if we have data that corresponds in some way and we don't want to lose the
      correlation we can still find out, which data objects are placed in a bin and do further analysis.
      For example consider data of the form 
      data[0] = list of different radiuses
      data[1] = list of (x,y) tuples
      So we create a histogram for data[0]
      To choose from which data set we want to form a histogram we specify position. So position=0
      creates a histogram for the list of radiuses.

      :param: list data: list of 2tuples where the first part of the tuple is of 
      dimension one. The 2nd part of the tuple can be anything.
      :returns: np.array n - Histogram of 
  """
  
  if (hrange is not None):
    mn, mx = hrange
    if mn > mx:
      raise AttributeError('max must be larger than min')
  
  data_array = [list() for _ in range(number_of_bins)]
  bins = np.linspace(mn, mx, number_of_bins+1)
  n = np.zeros(number_of_bins, int)
  block = 65536

  for i in np.arange(0, len(data), block):
    # sort the array for radius
    sa = np.sort(data[i:i+block], order='radius')

    # split radius and center data
    a,b = zip(*sa)

    # convert radius object to array and center object to list
    a = np.asarray(a)
    b = list(b)

    # intermediate calculation of the histogram VW used to add center entries to their
    # respective bins
    VW = np.r_[a.searchsorted(bins[:-1], 'left'), a.searchsorted(bins[-1], 'right')]
    VW = np.diff(VW)
    n += VW
    for counter,i in enumerate(VW):
      if i > 0:
        data_array[counter].append(b[:i])
        b = b[i:]
      else:
        continue


  return n, bins, data_array


def calculateCircleFromPoints(A, B, C):

  a = np.linalg.norm( C - B )
  b = np.linalg.norm( C - A )
  c = np.linalg.norm( B - A )
  a2 = a*a
  b2 = b*b
  c2 = c*c
  s = ( a + b + c ) / 2

  R = a * b * c / 4 / np.sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) )
  if R<3:
    b1 = a2 * ( b2 + c2 - a2 )
    b2 = b2 * ( a2 + c2 - b2 )
    b3 = c2 * ( a2 + b2 - c2 )
    P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( b1, b2, b3 ) ) )
    P /= b1 + b2 + b3
    return S_OK((R, P))
  else:
    return S_ERROR( "R > 3")

def main( combinationsList ):
  """ With the help of barycentric coordinates we calculate the radius and the center defined by each tuple of 3 points given as parameter

  @param: combinationsList: a list of all possible combinations of 3 points
  @returns: a center and radius dictionaries.
            center contains: - 2d histogram with the center (center['H'])
                             - xedges of the histogram (center['xedges'])
                             - yedges of the histogram (center['yedges'])
            radius contains: - 1d histogram of the radius
                             - xedges of the histogram

  """
  x = []
  y = []
  r = []
  xy = []
  for points in combinationsList:
    res = calculateCircleFromPoints(points[0], points[1], points[2])
    if not res['OK']:
      continue
    else:
      R, P = res['Value']
    xy.append( (P[0], P[1]))
    x.append( P[0] )
    y.append( P[1] )
    r.append( R )
  data = zip(r, xy)
  dtype = [('radius', float), ('center', tuple)]
  data = np.array(data,dtype=dtype)
  #center_histogram, c_xedges, c_yedges = np.histogram2d( x, y, NUMBER_OF_S_BINS, [[-1, 1], [-1, 1]] )
  radius_histogram, bins,  center_data = SHistogram(data, NUMBER_OF_R_BINS, [0,2])
  radius = {}
  radius['H'] = radius_histogram
  radius['xedges'] = bins
  radius['CenterData'] = center_data

  visualizeRadiusHistogram(radius)


def backgroundHistogram( filename ):
  """ Creates a histogram for both radius and center for a given filename. It is used for creating background histograms so we can minimize
      false hits from background and mismatched 3tuples (2 points of one circle and one of another e.g.)

      :param filename: path to the source text file with the background points
      :returns center dict and radius dict.

  """
  data = readFile2( filename )
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  center, radius = findCircles( combinationsList )
  
  
def extractCenter( center ):
  """ Simple method to find possible circle centers. Find highest entry in histogram save the value and set bin to 0 so we can look for the next.
      we do this <x> times.

      :param dict. center: center dictionary obtained by findCircles method.
      :returns list centers: a list with a tuple of x,y coordinates of possible centers

  """
  centers = []
  H = center['H']
  
  xedges = center['xedges']
  yedges = center['yedges']

  #TODO: to be able to set a mask to set values around the maximum index to 0
  # for example
  #                          000
  #                          0x0
  #                          000
  #
  # so x is the maximum we found and we want to set adjacent values to 0 as well
  while len(centers) < 10:
    index = np.argmax(H)
    xmax, ymax = np.unravel_index( index, (NUMBER_OF_S_BINS, NUMBER_OF_S_BINS) )
    centers.append((xedges[xmax], yedges[ymax]))
    #H[xmax][ymax] = 0

  return centers

def extractRadius( radius ):
  """ Simple method to find possible radiuses. Find highest entry in histogram, save the value and set bin to 0 and then look for the next.
      we do this <x> times.

      :param dict radius: radius dicitonary with 'H' histogram, 'xedges' and 'yedges'
      :returns list radius: a list with possible radiuses

  """
  radiuses = []
  H = radius['H']
  edges = radius['xedges']

  while len(radiuses) < 10:
    index = np.argmax(H)
    xmax = np.unravel_index( index, NUMBER_OF_R_BINS )
    radiuses.append(xmax)
    H[xmax] = 0
  return radiuses

def visualizeCenterHistogram( center ):
  xedges, yedges = center['xedges'], center['yedges']
  H = center['H']
  fig = plt.figure()
  plt.imshow(H, interpolation='nearest', origin='low',
                extent=[ xedges[0], xedges[-1], yedges[0], yedges[-1]])
  plt.colorbar()
  plt.show()
  
  
def visualizeRadiusHistogram( radius, backgroundHistogram=None ):
  step = 2./NUMBER_OF_R_BINS
  edges = np.arange(0,2,step)
  H = radius['H']
  if not type(backgroundHistogram) == type(None):
    H = H-backgroundHistogram['H']
  plt.bar(edges,H, width=step)
  plt.xlim(min(edges), max(edges))
  
  plt.show()

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'Please provide file to be read' )
  path = sys.argv[1]
  data = readFile2( path )
  timed = False
  if timed:
    with open( 'runtimes.txt', 'a' ) as f:
      f.write("Number of space bins: %s\n" % NUMBER_OF_S_BINS)
      f.write("Number of radius bins: %s\n" % NUMBER_OF_R_BINS)
      totalRuntime = 0
      with Timer() as t:
        combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
      f.write("=> elasped combinationsList: %s s\n" % t.secs)
      totalRuntime += t.secs

      with Timer() as t:
        center, radius = findCircles( combinationsList )
      f.write("=> elasped findCircles: %s s\n" % t.secs)
      totalRuntime += t.secs

      with Timer() as t:
        centers = extractCenter(center)
      f.write("=> elasped extractCenter: %s s\n" % t.secs)
      totalRuntime += t.secs

      with Timer() as t:
        radiuses = extractRadius(radius)
      f.write("=> elasped extractRadius: %s s\n" % t.secs)
      totalRuntime += t.secs
      f.write("Total Runtime: %s s\n" % totalRuntime)
      f.write("#############\n")
    f.close()

  else:
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    main( combinationsList )
  
    

