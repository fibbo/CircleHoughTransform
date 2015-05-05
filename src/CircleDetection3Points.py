'''
Created on Apr 29, 2015

@author: phi
'''
from Tools import readFile2
import sys
import itertools
import numpy as np
import time
import pdb


def findCircles( combinationsList ):
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
    if not R > 1:
      b1 = a ** 2 * ( b ** 2 + c ** 2 - a ** 2 )
      b2 = b ** 2 * ( a ** 2 + c ** 2 - b ** 2 )
      b3 = c ** 2 * ( a ** 2 + b ** 2 - c ** 2 )
      P = np.column_stack( ( A, B, C ) ).dot( np.hstack( ( b1, b2, b3 ) ) )
      P /= b1 + b2 + b3
      x.append( P[0] )
      y.append( P[1] )
      r.append( R )

  center_histogram = np.histogram2d( x, y, 100, [[-1, 1], [-1, 1]] )
  radius_histogram = np.histogram( r, 100, [0, 1] )
  return center_histogram, radius_histogram

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'please provide file to be read' )
  path = sys.argv[1]
  data = readFile2( path )
  start_time = time.time()
  combinationsList = list( itertools.combinations( data['allPoints'], 3 ) )
  findCircles( combinationsList )
  print time.time() - start_time
  print "The end"
