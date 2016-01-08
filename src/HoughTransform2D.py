from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from visualizeData import plotData
from Tools import *

DIMENSION = 200
VISUALISATION=True

def HoughTransform2D( data ):
  xbins = np.linspace(-0.5,0.5,DIMENSION)
  ybins = np.linspace(-0.5,0.5,DIMENSION)
  x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis,...] )

  for r in data['Radius']:
    weights = np.zeros( (DIMENSION,DIMENSION) )
    for x0,y0 in data['allPoints']:
      s = 0.001
      eta = (x-x0)**2 + (y-y0)**2 - r**2      
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
    if VISUALISATION:
      img = plt.imshow(weights, interpolation='nearest')
      plt.show()
    index = np.argmax(weights)
    i, j = np.unravel_index( index, (DIMENSION, DIMENSION))
    center = (xbins[i], ybins[j])
    print center
    used_xy = []
    used_xy = [tup for tup in data['allPoints'] if abs( ( tup[0] - center[0] ) ** 2 + ( tup[1] - center[1] ) ** 2 - r ** 2 ) < 2 * 0.005]
    data['allPoints'][:] = [tup for tup in data['allPoints'] if abs( ( tup[0] - center[0] ) ** 2 + ( tup[1] - center[1] ) ** 2 - r ** 2 ) > 2 * 0.005]




if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit( 'Please provide file to be read' )
  path = sys.argv[1]
  data = readFile(path)
  HoughTransform2D(data)
  for c in data['Center']:
    print c