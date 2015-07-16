'''
Created on Apr 29, 2015

@author: phi
'''
from Tools import readFile, S_ERROR, S_OK
import matplotlib.pyplot as plt
import sys
import itertools
import numpy as np
from timer import Timer

NUMBER_OF_R_BINS = 200 #bins for radius
NUMBER_OF_S_BINS = 1000 #bins for space
VISUALISATION = True

def SHistogram(data, number_of_bins, hrange=None):
  """ Creates a histogram for a given with number_of_bins entries within the boundaries given by range. 
      The important thing is that in addition to the histogram we also create an array that stores
      the single data objects that were filled in the histogram in the same bin they have been place
      in the histogram. So if we have data that corresponds in some way and we don't want to lose the
      correlation we can still find out, which data objects are placed in a bin and do further analysis.

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
  
  data_array = [None for _ in range(number_of_bins)]
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
        data_array[counter] = b[:i]
        b = b[i:]
      else:
        continue


  return S_OK( (n, bins, data_array) )


def calculateCircleFromPoints(combinationsList):
  r = []
  xy = []

  for points in combinationsList:
    A = points[0]
    B = points[1]
    C = points[2]
    a = np.linalg.norm( C - B )
    b = np.linalg.norm( C - A )
    c = np.linalg.norm( B - A )
    a2 = a*a
    b2 = b*b
    c2 = c*c
    s = ( a + b + c ) / 2

    R = a * b * c / 4 / np.sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) )
    #R = round(R, 4)
    #print R
    if R<3:
      lambda1 = a2 * ( b2 + c2 - a2 )
      lambda2 = b2 * ( a2 + c2 - b2 )
      lambda3 = c2 * ( a2 + b2 - c2 )
      P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( lambda1, lambda2, lambda3 ) ) )
      P /= lambda1 + lambda2 + lambda3
      xy.append( (P[0],P[1]) )
      r.append(R)

  return xy, r

def fullCenterHistogram( xy ):
  x,y = zip(*xy)
  H, xedges, yedges = np.histogram2d(x,y,NUMBER_OF_S_BINS,[[-1, 1], [-1, 1]])
  fig = plt.figure()
  plt.imshow(H, interpolation='nearest', origin='low',
                extent=[ xedges[0], xedges[-1], yedges[0], yedges[-1]])
  plt.colorbar()
  plt.show()

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

  xy, r = calculateCircleFromPoints( combinationsList )
  fullCenterHistogram( xy )
  data = zip(r, xy)
  dtype = [('radius', float), ('center', tuple)]
  data = np.array(data,dtype=dtype)
  res = SHistogram(data, NUMBER_OF_R_BINS, [0,2])
  if not res['OK']:
    return S_ERROR(res['Message'])
  
  radius_histogram, bins,  center_data = res['Value']

  # create a background histogram
  bkgHistogram, edges = backgroundHistogram('../data/left_to_right/2_0_circles_30_bg.txt')


  radius = {}
  radius['H'] = radius_histogram #- bkgHistogram
  radius['xedges'] = edges
  radius['center_data'] = center_data

  visualizeRadiusHistogram(radius)

  radiuses, center_data = extractRadius(radius)

  # check for each radius if we have a peak in center_data

  for radius, center in zip(radiuses, center_data):
    x,y = zip(*center)
    H, xedges, yedges = np.histogram2d(x,y,NUMBER_OF_S_BINS,[[-1, 1], [-1, 1]])
    center_dict = {}
    center_dict['H'] = H
    center_dict['xedges'] = xedges
    center_dict['yedges'] = yedges
    visualizeCenterHistogram(x,y)
    circle_center = extractCenter(center_dict)
    print radius, circle_center
  return S_OK()


def backgroundHistogram( filename ):
  """ Creates a histogram for both radius and center for a given filename. It is used for creating background histograms so we can minimize
      false hits from background and mismatched 3tuples (2 points of one circle and one of another e.g.)

      :param filename: path to the source text file with the background points
      :returns center dict and radius dict.
  """
  data = readFile( filename )
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  xy, r = calculateCircleFromPoints( combinationsList )
  bkgHistogram, edges = np.histogram(r, NUMBER_OF_R_BINS, [0,2])
  return bkgHistogram, edges
  
  
def extractCenter( center_dict ):
  """ Simple method to find possible circle centers. Find highest entry in histogram save the value and set bin to 0 so we can look for the next.
      we do this <x> times.

      :param dict. center: center dictionary obtained by findCircles method.
      :returns list centers: a list with a tuple of x,y coordinates of possible centers

  """

  H = center_dict['H']
  
  xedges = center_dict['xedges']
  yedges = center_dict['yedges']
  centers = []

  n = NUMBER_OF_S_BINS

  #TODO: to be able to set a mask to set values around the maximum index to 0
  # for example
  #                          000
  #                          0x0
  #                          000
  #
  # so x is the maximum we found and we want to set adjacent values to 0 as well
  while True:
    index = np.argmax(H)
    i, j = np.unravel_index( index, (NUMBER_OF_S_BINS, NUMBER_OF_S_BINS) )
    n_entries =   sum(sum(H[i-1 if i>0 else i:i+2 if i<n else i+1,j-1 if j>0 else j:j+2 if j<n else j+1]))

    if n_entries < 30:
      break

    # Set the entries to 0 so they won't contribute twice
    i_index = range(i-1 if i>0 else i,i+2 if i<n else i+1)
    j_index = range(j-1 if j>0 else j,j+2 if j<n else j+1)
    for i in i_index:
      for j in j_index:
        H[i][j] = 0
    
    centers.append( (xedges[i], yedges[j]) )

  return centers

def extractRadius( radius_dict ):
  """ Simple method to find possible radiuses. Find highest entry in histogram, save the value and set bin to 0 and then look for the next.


      :param dict radius: radius dicitonary with 'H' histogram, 'xedges' and 'yedges'
      :returns radius, center_data
  """
  radiuses = []
  center_data = []
  H = radius_dict['H']
  edges = radius_dict['xedges']
  center = radius_dict['center_data']
  while True:
    i = np.argmax(H)
    n = NUMBER_OF_R_BINS
    n_entries = sum(H[i-1 if i>0 else i:i+2 if i<n else i+1])
    if n_entries < 200:
      # there are less than 200 entries in 3 bins
      break

    radiuses.append(edges[i])
    center_data.append(center[i])
    index_list = range(i-1 if i>0 else i,i+2 if i<n else i+ 1)
    for index in index_list:
      H[index] = 0
  return radiuses, center_data 

def visualizeCenterHistogram( x,y ):
  if not VISUALISATION:
    pass
  else:
    H, xedges, yedges, _ = plt.hist2d(x,y,NUMBER_OF_S_BINS, [[-1,1],[-1,1]])
    plt.show()
  
  
def visualizeRadiusHistogram( radius ):
  if not VISUALISATION:
    pass
  else:
    step = 2./NUMBER_OF_R_BINS
    edges = np.arange(0,2,step)
    H = radius['H']
    plt.bar(edges,H, width=step)
    plt.xlim(min(edges), max(edges))
    
    plt.show()

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'Please provide file to be read' )
  path = sys.argv[1]
  data = readFile( path )
  timed = False
  import pdb; pdb.set_trace()
  combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
  res = main( combinationsList )
  if not res['OK']:
    print res['Message']
  else:
    print 'Done'
  
    

