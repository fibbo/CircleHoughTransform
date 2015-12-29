from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from visualizeData import plotData
from Tools import *

DIMENSION = 1001

def HoughTransform2D( data ):
  xbins = np.linspace(-0.5,0.5,DIMENSION)
  ybins = np.linspace(-0.5,0.5,DIMENSION)
  x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis,...] )

  for r in data['Radius']:
    weights = np.zeros( (DIMENSION,DIMENSION) )
    for x0,y0 in data['allPoints']:
      s = 2*0.001
      eta = (x-x0)**2 + (y-y0)**2 - r**2      
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
    index = np.argmax(weights)
    i, j = np.unravel_index( index, (DIMENSION, DIMENSION))
    n = DIMENSION
    i_index = range(i-1 if i>0 else i,i+2 if i<n else i+1)
    j_index = range(j-1 if j>0 else j,j+2 if j<n else j+1)
    for ii in i_index:
      weights[ii][j] = 0
    for jj in j_index:
      weights[i][jj] = 0
    print xbins[i], ybins[j]




if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit( 'Please provide file to be read' )
  path = sys.argv[1]
  data = readFile(path)
  HoughTransform2D(data)
  for c in data['Center']:
    print c