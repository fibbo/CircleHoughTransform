# default python imports
from math import cos, sin, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from Histogram import Histogram
from Histogram2D import Histogram2D
from Tools import *



def HoughTransform( data ):

  listOfHistograms = createHistograms( data )
  listOfRadiuses = findRadiusClusters( listOfHistograms )
  plt.scatter( data['x'], data['y'] )
  # import pdb; pdb.set_trace()
  fig = plt.gcf()
  for radius in listOfRadiuses:
    if not radius['Radius']:
      continue

    for i in radius['Radius']:
      circle = plt.Circle( radius['Center'], i, fill = False )
      fig.gca().add_artist( circle )
  colors = ['red', 'green', 'blue', 'yellow']
  i = 0
  for center, radius in zip(data['Center'], data['Radius']):
    fig.gca().add_artist( plt.Circle( center, radius, fill = False, color = colors[i] ) )
    i+=1
  
  plt.show()

def findRadiusClusters( histograms ):
  """ search through the histograms for a cluster. when a cluster is found we save the radius value
      of that cluster in a list and return it.
  :param Histogram histograms: list of histograms
  :returns list of dicts for the different histograms with 'Center' and 'Radius' key.
  """

  res = []
  for h in histograms:
    clusters = []
    temp = []
    j = 0
    for entry in h.hist:
      if entry > 2:
        clusters.append( j )
      j += 1
    for cid in clusters:
      temp.append( h.getValue( cid ) )
    result = {}
    result['Radius'] = temp
    result['Center'] = h.center
    res.append( result )

  return res

def calcCircles( data, r ):
  step = 2 * sconst.pi / 100
  a = []
  b = []
  thetas = np.arange( 0, 2 * sconst.pi, step )

  for x, y in zip( data['x'], data['y'] ):
    for theta in thetas:
      a.append( x - r * cos( theta ) )
      b.append( y - r * sin( theta ) )

  return { 'a' : a, 'b' : b}

def create2DHistograms( data ):
  histograms = []
  for r in data['Radius']:
    res = calcCircles( data, r )
    plt.scatter( res['a'], res['b'] )
    plt.show()


  return histograms

def createHistograms( data ):
  """ Fill a different histogram for circle centers with the distance between all points and the
      respective center of the circle.
      :param double x, list of x coordinates
      :param double y, list of y coordinates
      :param list circle_centers, list of circle centers
      :returns a list of histograms. one for each circle center
  """

  histograms = []
  for a, b in data['Center']:
    h = Histogram( 0.001, 0.29, 1.1, ( a, b ) )
    res = calcRadius( data, a, b )
    for radius in res['Radius']:
      h.addValue( radius )
    histograms.append( h )
  return histograms

def calcRadius( data, a, b ):
  res = { 'Radius' : [] }
  x = data['x']
  y = data['y']
  for i in range( len( x ) ):
      radius = sqrt( ( x[i] - a ) ** 2 + ( y[i] - b ) ** 2 )
      res['Radius'].append( radius )

  return res

def find2DCluster( data, neighbor_weights = False, weight_factor = 0.5 ):
  """ Goes through the weight matrix and finds the highest value or the highest cluster of values.
      If neighbor_weights is false just the matrix point itself is considered. If neighbor_weights
      is True the top, bottom, left and right neighbor are considered as well with the weight_factor

      In the end the grid point with the highest sum is saved and considered being the center of the
      circle

      :param dict data: data dict with the 'Results' key which contains the weight matrices
      :param returns: adds 'CalcCenter' key to the data dict with the found centers for all
                      weight matrices

  """
  centers = []
  gridSumValue = []
  for w in data['Results']:
    max_value = 0.
    it = np.nditer( w, flags = ['multi_index'] )
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
    centers.append( ( data['xbins'][imax], data['ybins'][jmax] ) )
    gridSumValue.append( gridSum )

  data['CalcCenter'] = centers
  data['GridSum'] = gridSumValue

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
  x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis, ...] )
  for r in data['Radius']:
    w = np.zeros( ( dim, dim ) )
    res = weight_function( w, r, x, y, data )


    results.append( res )
  print xbins, ybins
  data['xbins'] = xbins
  data['ybins'] = ybins
  data['Results'] = results


def visualize( data ):
  """ Visualizing the data. First a scatter plot of the simulated data then the circles found by the algorithm.
      for comparison the real circles are plotted as well

      :param dict data. data holds all the information
  """

  plt.scatter( data['x'], data['y'] )
  fig = plt.gcf()
  # ax1 = plt.subplot( 1, 1, 1 )
  # x0, x1 = ax1.get_xlim()
  # y0, y1 = ax1.get_ylim()
  # print ax1
  colors = ['red', 'blue', 'green', 'yellow', 'purple']

  for c, r in zip( data['CalcCenter'], data['Radius'] ):
    fig.gca().add_artist( plt.Circle( ( c[0], c[1] ), r, fill = False, color = 'grey' ) )

  i = 0
  for c1, c2 in data['Center']:
    fig.gca().add_artist( plt.Circle( ( c1, c2 ), data['Radius'][i], fill = False, color = colors[i] ) )
    i += 1

  plt.show()


if __name__ == '__main__':

#### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'please provide file to be read' )
  path = sys.argv[1]
  data = readFile( path )

#### run specific methods ####
  calculateWeights( data, gaussWeight, dim = 200 )
  find2DCluster( data, neighbor_weights = False, weight_factor = 0.5 )


  colors = ['red', 'blue', 'green', 'yellow', 'purple']
  i = 0
  for c1, c2 in data['CalcCenter']:
    print "(%s, %s) - %s" % ( c1, c2, colors[i] )
    i += 1
  # print data['GridSum']
  #visualize( data )
  HoughTransform(data)
  print 'The End'



  # h.printHistogramVariables()
