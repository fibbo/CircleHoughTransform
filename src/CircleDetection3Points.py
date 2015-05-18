'''
Created on Apr 29, 2015

@author: phi
'''
from Tools import readFile2
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import itertools
import numpy as np
from timer import Timer

NUMBER_OF_R_BINS = 1000 #bins for radius
NUMBER_OF_S_BINS = 2000 #bins for space

def findCircles( combinationsList ):
#  data = []
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
    if R <= 1.:
      b1 = a ** 2 * ( b ** 2 + c ** 2 - a ** 2 )
      b2 = b ** 2 * ( a ** 2 + c ** 2 - b ** 2 )
      b3 = c ** 2 * ( a ** 2 + b ** 2 - c ** 2 )
      P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( b1, b2, b3 ) ) )
      P /= b1 + b2 + b3
      x.append( P[0] )
      y.append( P[1] )
      r.append( R )
  
  center_histogram, c_xedges, c_yedges = np.histogram2d( x, y, NUMBER_OF_S_BINS, [[-1, 1], [-1, 1]] )
  radius_histogram, r_xedges = np.histogram( r, NUMBER_OF_R_BINS, [0, 1], normed=True )
  center = {}
  center['H'] = center_histogram
  center['xedges'] = c_xedges
  center['yedges'] = c_yedges
  radius = {}
  radius['H'] = radius_histogram
  radius['xedges'] = r_xedges

  return center, radius

def extractCenter( center ):
  centers = []
  H = center['H']
  
  xedges = center['xedges']
  yedges = center['yedges']
  while len(centers) < 8:
    index = np.argmax(H)
    xmax, ymax = np.unravel_index( index, (NUMBER_OF_S_BINS, NUMBER_OF_S_BINS) )
    centers.append((xedges[xmax], yedges[ymax]))
    H[xmax][ymax] = 0

  return centers

def extractRadius( radius ):
  radiuses = []
  H = radius['H']
  edges = radius['xedges']

  while len(radiuses) < 8:
    index = np.argmax(H)
    xmax = np.unravel_index( index, NUMBER_OF_R_BINS )
    radiuses.append(edges[xmax])
    H[xmax] = 0
  return radiuses

def visualizeCenterHistogram( center ):
  xedges, yedges = center['xedges'], center['yedges']
  H = center['H']
  fig = plt.figure()
  #ax = fig.add_subplot(111)
  plt.imshow(H, interpolation='nearest', origin='low',
                extent=[ xedges[0], xedges[-1], yedges[0], yedges[-1]])
  plt.colorbar()
  ax = fig.add_subplot(121)
  ax.set_title('pcolormesh: exact bin edges')
  X, Y = np.meshgrid(xedges, yedges)
  #ax.pcolormesh(X, Y, H)
  ax.set_aspect('equal')
  
  
def visualizeRadiusHistogram( radius ):
  step = 1./NUMBER_OF_R_BINS
  edges = np.arange(0,1,step)
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
#  visualizeRadiusHistogram(radius)

  #visualize(center, radius)
