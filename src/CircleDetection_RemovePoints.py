# default python imports
from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb
import sys


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
  plt.ylim( -2, 1 )
  plt.xlim( -2, 1 )
  plt.scatter( data['x'], data['y'] )
  fig = plt.gcf()

  colors = ['red', 'blue', 'green', 'yellow', 'purple']

  for c, r in zip( data['Results'], data['Radius'] ):
    fig.gca().add_artist( plt.Circle( ( c[0], c[1] ), r, fill = False, color = 'grey' ) )

  i = 0
  for c1, c2 in data['Center']:
    fig.gca().add_artist( plt.Circle( ( c1, c2 ), data['Radius'][i], fill = False, color = colors[i] ) )
    i += 1

  plt.show()

def findCircles( data, weight_function, dim = 100 ):
  """ Define the plane, bin size and create an x and y array that are used for weight calculations.
  Iterate through the data['Radius'] to find a suitable circle for the given radius.
  The Circle finding goes through 4 steps:
  1. Calculate a weight for each bin entry, meaning how likely it is for that bin to be a center of a circle for a given radius
  2. Find the highest value in the weight matrix - this is the center for the circle
  3. Remove data points data['x'] and data['y'] that lie on this circle
  4. Add the center point to the results list

  :param dict data: Contains radius, x and y information. We fill it with xbins and ybins vectors and weight matrices
  :param func weight_function: this is a function that is used to calculate the weight of a bin entry
  :param int dim: Dimension of the histrogram per dimension
  :returns Nothing, but adds the result list to the data object.
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
  """ Given a center point we found we remove points that are on those circle.

  :params float tuple center: (cx, cy) x and y coordinate of the center
  :params dict data: contains the data points

  """
  i = 0
  used_x = []
  used_y = []
  for x0, y0 in zip( data['x'], data['y'] ):
    if ( abs( ( x0 - center[0] ) ** 2 + ( y0 - center[1] ) ** 2 - r ** 2 ) ) < 0.01:
      used_x.append( data['x'].pop( i ) )
      used_y.append( data['y'].pop( i ) )
    else:
      i += 1

  return zip( used_x, used_y )

def findCenter( w, data, neighbor_weights = False, weight_factor = 0.5 ):
  """ Goes through the weight matrix and finds the highest value or the highest cluster of values.
      If neighbor_weights is false just the matrix point itself is considered. If neighbor_weights
      is True the top, bottom, left and right neighbor are considered as well with the weight_factor

      In the end the grid point with the highest sum is saved and considered being the center of the
      circle

      :param dict data: data dict with the 'Results' key which contains the weight matrices
      :param returns: (cx, cy) a tuple with the x and y coordinate of the found center.

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

  x, y = list( data['x'] ), list( data['y'] )
  # shuffle the order the radiuses - algorithm shouldnt depend on the order
  # which it does at the moment - also shuffling the centers with the radiuses
  # or during the visualisation wrong centers get match with wrong radiuses
  combined = zip( data['Center'], data['Radius'], data['nPoints'] )
  combined = sorted( combined, key = lambda x: x[2], reverse = True )

  data['Center'][:], data['Radius'][:], data['nPoints'] = zip( *combined )

#### run specific methods ####
  findCircles( data, gaussWeight, 80 )

  # restore initial data points so the plot includes all data points (noise included)
  data['x'], data['y'] = x, y

  visualize( data )


  print 'The End'

