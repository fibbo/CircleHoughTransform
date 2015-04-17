# default python imports
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb
import sys
import random

from Tools import readFile

def inverseWeight( weight_matrix, radius, x, y, data ):
  for x0, y0 in zip( data['x'], data['y'] ):
    weight_matrix += 1.0 / ( np.abs( ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - radius ** 2 ) + 10e-3 )
  return weight_matrix

def gaussWeight( weight_matrix, radius, x, y, data ):
  for x0, y0 in zip( data['x'], data['y'] ):
    s = 2 * data['bin_width']
    eta = ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - radius ** 2
    weight_matrix += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
  return weight_matrix

def visualize( data ):
  """ Visualizing the data. First a scatter plot of the simulated data then the circles found by the algorithm.
      for comparison the real circles are plotted as well

      :param dict data. data holds all the information
  """

  plt.scatter( data['x'], data['y'] )
  fig = plt.gcf()
#   ax1 = plt.subplot( 1, 1, 1 )
#   x0, x1 = ax1.get_xlim()
#   y0, y1 = ax1.get_ylim()
#   print ax1
  colors = ['red', 'blue', 'green', 'yellow', 'purple']

  for c, r in zip( data['Results'], data['Radius'] ):
    fig.gca().add_artist( plt.Circle( ( c[0], c[1] ), r, fill = False, color = 'grey' ) )

  i = 0
  for c1, c2 in data['Center']:
    fig.gca().add_artist( plt.Circle( ( c1, c2 ), data['Radius'][i], fill = False, color = colors[i] ) )
    i += 1

  plt.show()

def calculateWeights( data, weight_function, dim = 100 ):
  """ We create a histogram from xmin to xmax and ymin to ymax and calculate then the distances from the grid
      points to the input points. if a grid point has the same distance to several input points it could be that
      this grid point is the center of a circle on which the input points lie

  :param dict data: Contains radius, x and y information. We fill it with xbins and ybins vectors and weight matrices
  """
  results = []
  xmin = -1.0
  xmax = 1.0
  ymin = -1.0
  ymax = 1.0
  # divide the x and y axis in [steps] equal parts from 0 to 1
  i = ( np.arange( 0, 100, 100. / dim ) + 0.5 ) / 100
  data['bin_width'] = 1. / dim
  xbins = xmin + i * ( xmax - xmin )
  ybins = ymin + i * ( ymax - ymin )
  data['xbins'] = xbins
  data['ybins'] = ybins
  data['Results'] = results
  x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis, ...] )
  for r in data['Radius']:
    w = np.zeros( ( dim, dim ) )
    w = weight_function( w, r, x, y, data )
    center = findCenter( w, data )
    removePoints( center, data, r )
    results.append( center )



def removePoints( center, data, r ):
  i = 0
  used_x = []
  used_y = []
  for x0, y0 in zip( data['x'], data['y'] ):
    if ( abs( ( x0 - center[0] ) ** 2 + ( y0 - center[1] ) ** 2 - r ** 2 ) ) < 0.01:
      used_x.append( data['x'].pop( i ) )
      used_y.append( data['y'].pop( i ) )
    else:
      i += 1

def findCenter( w, data, neighbor_weights = False, weight_factor = 0.5 ):
  """ Goes through the weight matrix and finds the highest value or the highest cluster of values.
      If neighbor_weights is false just the matrix point itself is considered. If neighbor_weights
      is True the top, bottom, left and right neighbor are considered as well with the weight_factor

      In the end the grid point with the highest sum is saved and considered being the center of the
      circle

      :param dict data: data dict with the 'Results' key which contains the weight matrices
      :param returns: adds 'CalcCenter' key to the data dict with the found centers for all
                      weight matrices

  """
  max_value = 0.
  it = np.nditer( w, flags = ['multi_index'] )
  imax, jmax = 0, 0
  while not it.finished:
    gridSum = 0
    if neighbor_weights:
      i, j = it.multi_index
      if not i - 1 < 0:
        gridSum += weight_factor * w[i - 1][j]
      if not j - 1 < 0:
        gridSum += weight_factor * w[i][j - 1]
      if not i + 1 >= len( w ):
        gridSum += weight_factor * w[i + 1][j]
      if not j + 1 >= len( w ):
        gridSum += weight_factor * w[i][j + 1]
      gridSum += it[0]
      if gridSum > max_value:
        max_value = gridSum
        imax, jmax = i, j
    else:
      gridSum = it[0]
    if gridSum > max_value:
      max_value = it[0]
      imax, jmax = it.multi_index
    it.iternext()
  center = ( data['xbins'][imax], data['ybins'][jmax] )

  return center


if __name__ == '__main__':

#### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'please provide file to be read' )
  path = sys.argv[1]
  data = readFile( path )

  # saving the data points since some points get removed in  the algorithm
  x, y = data['x'], data['y']

  # shuffle the order the radiuses - algorithm shouldnt depend on the order
  # which it does at the moment - also shuffling the centers with the radiuses
  # or during the visualisation wrong centers get match with wrong radiuses
  combined = zip( data['Center'], data['Radius'] )
  random.shuffle( combined )
  data['Center'][:], data['Radius'][:] = zip( *combined )

#### run specific methods ####
  calculateWeights( data, gaussWeight, 300 )

  # restore initial data points so the plot includes all data points (noise included)
  data['x'], data['y'] = x, y

  visualize( data )


  print 'The End'

