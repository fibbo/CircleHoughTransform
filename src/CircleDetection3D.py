# default python imports
from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import sys
import time


from Tools import readFile

def inverseWeight( weight_matrix, radius, x, y, data ):
  for x0, y0 in zip( data['x'], data['y'] ):
    weight_matrix += 1.0 / ( np.abs( ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - radius ** 2 ) + 10e-3 )
  return weight_matrix

def gaussWeight( weight_matrix, r, x, y, data ):
  for x0, y0 in zip( data['x'], data['y'] ):
    s = 2 * data['bin_width']
    eta = ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - r ** 2
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
  print 'Number of circles found %s' % len( data['CenterCalc'] )
  for c, r in zip( data['CenterCalc'], data['RadiusCalc'] ):
    fig.gca().add_artist( plt.Circle( ( c[0], c[1] ), r, fill = False, color = 'grey' ) )

  i = 0
  for c1, c2 in data['Center']:
    fig.gca().add_artist( plt.Circle( ( c1, c2 ), data['Radius'][i], fill = False, color = colors[i] ) )
    i += 1

  plt.show()

def findCircles( data, weight_function, space_dim = 100, r_dim = 100 ):
  """ Define the plane, bin size and create an x and y array that are used for weight calculations.
  Iterate through the data['Radius'] to find a suitable circle for the given radius.
  The Circle finding goes through 4 steps:
  1. Calculate a weight for each bin entry. W is a 3 dimensional array where the first 2 dimensions correspond to the space
     and the 3rd dimension is for different radiuses
  2. Find the highest value in the weight matrix - [0] and [1] are the x and y coordinate for the center [2] the radius
  3. Remove data points data['x'] and data['y'] that lie on this circle
  4. Add the center point and radius to the results list

  :param dict data: Contains radius, x and y information. We fill it with xbins and ybins vectors and weight matrices
  :param func weight_function: this is a function that is used to calculate the weight of a bin entry
  :param int space_dim: Size of each of the space arrays
  :param int r_dim: Size of the radius array
  :returns Nothing, but adds the result list to the data object.
  """
  centers = []
  radius = []
  xmin = -1.0
  xmax = 1.0
  ymin = -1.0
  ymax = 1.0
  rmin = 0.0
  rmax = 1
  # divide the x and y axis in [steps] equal parts from 0 to 1
  i = ( np.arange( 0, 100, 100. / space_dim ) + 0.5 ) / 100
  # divide r in [steps] equal parts from 0 to 1
  ii = ( np.arange( 0, 100, 100. / r_dim ) + 0.5 ) / 100
  data['space_dim'] = space_dim
  data['r_dim'] = r_dim
  data['bin_width'] = ( xmax - xmin ) / space_dim
  xbins = xmin + i * ( xmax - xmin )
  ybins = ymin + i * ( ymax - ymin )
  rbins = rmin + ii * ( rmax - rmin )
  data['xbins'] = xbins
  data['ybins'] = ybins
  data['rbins'] = rbins
  data['CenterCalc'] = centers
  data['RadiusCalc'] = radius
  x, y, r = np.broadcast_arrays( xbins[..., np.newaxis, np.newaxis], ybins[np.newaxis, ..., np.newaxis], rbins[np.newaxis, np.newaxis, ...] )
  max_value = 140
  while max_value >= 140:
    w = np.zeros( ( space_dim, space_dim, r_dim ) )
    w = weight_function( w, r, x, y, data )
    center, rad, max_value = findCenter( w, data )
    removePoints( center, data, rad )
    centers.append( center )
    radius.append( rad )


def removePoints( center, data, r ):
  """ Given a center point we found we remove points that are on those circle.

  :params float tuple center: (cx, cy) x and y coordinate of the center
  :params dict data: contains the data points

  """
  i = 0
  used_x = []
  used_y = []
  for x0, y0 in zip( data['x'], data['y'] ):
    if ( abs( ( x0 - center[0] ) ** 2 + ( y0 - center[1] ) ** 2 - r ** 2 ) ) < 2 * data['bin_width']:
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
  if not neighbor_weights:
    index = np.argmax( w )
    imax, jmax, rmax = np.unravel_index( index, ( data['space_dim'], data['space_dim'], data['r_dim'] ) )
    max_value = w[imax][jmax][rmax]
  center = ( data['xbins'][imax], data['ybins'][jmax] )
  radius = data['rbins'][rmax]
  return center, radius, max_value


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
  start_time = time.time()
  findCircles( data, gaussWeight, space_dim = 80, r_dim = 100 )

  # restore initial data points so the plot includes all data points (noise included)
  # data['x'], data['y'] = x, y
  print time.time() - start_time
  visualize( data )


  print 'The End'

