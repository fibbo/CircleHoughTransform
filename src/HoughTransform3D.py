from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from visualizeData import plotData
from Tools import *

DIMENSION=200
R_DIMENSION=200

def HoughTransform3D( data ):
  xbins = np.linspace(-0.5,0.5,DIMENSION)
  ybins = np.linspace(-0.5,0.5,DIMENSION)
  rbins = np.linspace(0,0.5, R_DIMENSION)

  x,y,r = np.broadcast_arrays( xbins[...,np.newaxis,np.newaxis], \
                               ybins[np.newaxis,...,np.newaxis], \
                               rbins[np.newaxis,np.newaxis,...])

  score = 2000
  used_xy = []
  again_xy = []
  while True:
    # print score
    weights = np.zeros( (DIMENSION, DIMENSION, R_DIMENSION))
    for x0,y0 in data['allPoints']:
      s = 2*0.005
      eta = ( x - x0 ) ** 2 + ( y - y0 ) ** 2 - r ** 2
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 )\
                                                              / ( 2 * s ** 2 ) )
    index = np.argmax( weights )
    ii,jj,rr = np.unravel_index( index, (DIMENSION, DIMENSION, R_DIMENSION))
    score = weights[ii][jj][rr]
    if score < 100:
      break
    center = (xbins[ii], ybins[jj])
    radius = rbins[rr]
    i = 0
    used_xy = []
    used_xy = [tup for tup in data['allPoints'] if abs( ( tup[0] - center[0] ) ** 2 + ( tup[1] - center[1] ) ** 2 - radius ** 2 ) < 2 * 0.005]
    data['allPoints'][:] = [tup for tup in data['allPoints'] if abs( ( tup[0] - center[0] ) ** 2 + ( tup[1] - center[1] ) ** 2 - radius ** 2 ) > 2 * 0.005]
    ux, uy = zip(*used_xy)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.scatter(ux, uy,c='r')
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    if data['allPoints']:
      ox, oy = zip(*data['allPoints'])
      ax2 = fig.add_subplot(212)
      ax2.scatter(ox, oy)
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    plt.show()
    print center, radius




if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit("Please provide file to be read")

  path = sys.argv[1]
  data = readFile(path)
  HoughTransform3D(data)