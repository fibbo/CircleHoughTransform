'''
Created on Apr 29, 2015

@author: phi
'''
from Tools import readFile2
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import pdb
import itertools
import numpy as np
from timer import Timer

NUMBER_OF_R_BINS = 800 #bins for radius
NUMBER_OF_S_BINS = 2000 #bins for space

def findCircles( combinationsList ):
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
  for points in combinationsList:
    A = points[0]
    B = points[1]
    C = points[2]

    a = np.linalg.norm( C - B )
    b = np.linalg.norm( C - A )
    c = np.linalg.norm( B - A )

    s = ( a + b + c ) / 2

    R = a * b * c / 4 / np.sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) )
    if True:
      b1 = a ** 2 * ( b ** 2 + c ** 2 - a ** 2 )
      b2 = b ** 2 * ( a ** 2 + c ** 2 - b ** 2 )
      b3 = c ** 2 * ( a ** 2 + b ** 2 - c ** 2 )
      P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( b1, b2, b3 ) ) )
      P /= b1 + b2 + b3
      x.append( P[0] )
      y.append( P[1] )
      r.append( R )
  
  center_histogram, c_xedges, c_yedges = np.histogram2d( x, y, NUMBER_OF_S_BINS, [[-1, 1], [-1, 1]] )
  radius_histogram, r_xedges = np.histogram( r, NUMBER_OF_R_BINS, [0, 2], normed=False )
  center = {}
  center['H'] = center_histogram
  center['xedges'] = c_xedges
  center['yedges'] = c_yedges
  radius = {}
  radius['H'] = radius_histogram
  radius['xedges'] = r_xedges

  return center, radius

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
    radiuses.append(edges[xmax])
    #H[xmax] = 0
  return radiuses

def visualizeCenterHistogram( center ):
  xedges, yedges = center['xedges'], center['yedges']
  H = center['H']
  fig = plt.figure()
  plt.imshow(H, interpolation='nearest', origin='low',
                extent=[ xedges[0], xedges[-1], yedges[0], yedges[-1]])
  plt.colorbar()
  plt.show()
  
  
def visualizeRadiusHistogram( radius ):
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
  data = readFile2( path )
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
  print centers
  print radiuses
  visualizeRadiusHistogram(radius)

  visualizeCenterHistogram(center)
