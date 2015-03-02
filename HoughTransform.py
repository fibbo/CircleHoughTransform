#default python imports
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from Histogram import Histogram
from Histogram2D import Histogram2D
from Tools import *



def HoughTransform( data ):

  histograms = createHistograms( data )
  radiuses = findRadiusClusters( histograms )
  plt.scatter( data['x'], data['y'] )

  fig = plt.gcf()
  for radius in radiuses:
    # pdb.set_trace()
    if not radius['Radius']:
      continue

    for i in radius['Radius']:
      circle = plt.Circle( radius['Center'], i, fill = False )
      fig.gca().add_artist( circle )

  fig.gca().add_artist( plt.Circle( ( -0.342, -0.994 ), 0.719, fill = False, color = 'red' ) )
  fig.gca().add_artist( plt.Circle( ( -0.821, -0.656 ), 0.563, fill = False, color = 'green' ) )
  fig.gca().add_artist( plt.Circle( ( -0.261, -0.328 ), 0.892, fill = False, color = 'blue' ) )
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
      if entry > 5:
        clusters.append( j )
      j += 1
    for cid in clusters:
      temp.append( h.getValue( cid ) )
    result = {}
    result['Radius'] = temp
    result['Center'] = h.center
    res.append( result )

  return res

def HoughTransform2D( data ):
  histograms = create2DHistograms( data )
  for h in histograms:
    h.printHistogramVariables()()

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

def createHistograms(data):
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
      h.addValue(radius)
    histograms.append(h)
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
  print '###########'
  for w in data['Results']:
    max_value = 0.
    it = np.nditer( w, flags = ['multi_index'] )
    while not it.finished:
      if neighbor_weights:
        gridSum = 0
        i, j = it.multi_index
        if not i - 1 < 0:
          gridSum += weight_factor * w[i - 1][j]
        if not j - 1 < 0:
          gridSum += weight_factor * w[i][j - 1]
        if not i + 1 >= len( w ):
          gridSum += weight_factor * w[i + 1][j]
        if not j + 1 >= len( w ):
          gridSum += weight_factor * w[i][j + 1]
        gridSum += 2 * weight_factor * it[0]
      else:
        gridSum = it[0]
        print it[0]
      if gridSum > max_value:
        max_value = it[0]
        i, j = it.multi_index
      it.iternext()
    print '#################'
    centers.append( ( data['xbins'][i], data['ybins'][j] ) )
    gridSumValue.append( gridSum )

  data['CalcCenter'] = centers
  data['GridSum'] = gridSumValue





def calculateWeights( data, dim = 100 ):
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
  xbins = xmin + i * ( xmax - xmin )
  ybins = ymin + i * ( ymax - ymin )
  x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis, ...] )
  for r in data['Radius']:
    w = np.zeros( ( dim, dim ) )
    for x0, y0 in zip( data['x'], data['y'] ):
      w += 1.0 / ( np.abs( ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - r ** 2 ) + 10e-6 )

    results.append( w )
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
  for c, r in zip( data['CalcCenter'], data['Radius'] ):
    fig.gca().add_artist( plt.Circle( ( c[0], c[1] ), r, fill = False, color = 'red' ) )
  fig.gca().add_artist( plt.Circle( ( data['Center'][2][0], data['Center'][2][1] ), data['Radius'][2], fill = False, color = 'black' ) )
  fig.gca().add_artist( plt.Circle( ( data['Center'][0][0], data['Center'][0][1] ), data['Radius'][0], fill = False, color = 'black' ) )
  fig.gca().add_artist( plt.Circle( ( data['Center'][1][0], data['Center'][1][1] ), data['Radius'][1], fill = False, color = 'black' ) )

  plt.show()


if __name__ == '__main__':

#### read data #####
  if len(sys.argv) < 2:
    sys.exit( 'please provide file to be read' )
  path = sys.argv[1]
  data = readFile(path)

#### run spefici methods ####

  calculateWeights( data, dim = 100 )
  find2DCluster( data, neighbor_weights = False, weight_factor = 0.5 )
  print data['CalcCenter']
  print data['GridSum']
  visualize( data )

  print 'The End'
  

  
  #h.printHistogramVariables()
