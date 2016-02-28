from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb
from random import shuffle

from visualizeData import plotData
from Tools import *

DIMENSION = 1001
VISUALISATION=True
PLOT_PROJECTION=False

font = {'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def HoughTransform2D( data, name ):
  x = np.linspace(-0.5,0.5,DIMENSION)
  y = np.linspace(-0.5,0.5,DIMENSION)
  # x, y = np.broadcast_arrays( xbins[..., np.newaxis], ybins[np.newaxis,...] )
  counter = 1
  circles = []
  used_xy = []
  s = 0.001
  combined = zip(data['Radius'], data['Center'])
  shuffle(combined)
  data['Radius'][:], data['Center'][:] = zip(*combined)
  for r in data['Radius']:
    print r
    weights = np.zeros( (DIMENSION,DIMENSION) )
    for x0,y0 in data['allPoints']:
      eta = (x[None,:]-x0)**2 + (y[:,None]-y0)**2 - r**2      
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
    if VISUALISATION:
      img = plt.imshow(weights, interpolation='nearest', origin='lower')
      cba = plt.colorbar(img)
      plt.xlabel('$x$')
      plt.ylabel('$y$')
      plt.savefig('../img/2D_HT/center_scores_%s_%s_newsigma.pdf' % (name, counter))
      plt.close()
    index = np.argmax(weights)
    i, j = np.unravel_index( index, (DIMENSION, DIMENSION))

    if PLOT_PROJECTION:
      for slice_index in range(i-4,i+5,4):
        plt.plot(x, weights[:][slice_index])
        plt.xlim(-0.5,0.5)
        # plt.close()
        plt.savefig('../img/2D_HT/projection/stacked_projection_%s_%s.pdf' % ((slice_index-i+5), counter))


    print "Score: %s - (%s, %s)" % (weights[i,j],x[j],y[i])
    clearRange = 20
    for xx in range(i-clearRange,i+clearRange):
      for yy in range(j-clearRange,j+clearRange):
        weights[xx,yy] = 0

    index2 = np.argmax(weights)
    ii, jj = np.unravel_index( index2, (DIMENSION, DIMENSION))
    print "2nd highest score: %s - (%s, %s)" % (weights[ii,jj],x[jj],y[ii])


    circle = {}
    circle['center'] = (x[j], y[i])
    circle['radius'] = r
    circles.append(circle)
    counter += 1
  #   allPoints = []
  #   while len(data['allPoints']):
  #     tup = data['allPoints'].pop()
  #     if abs( ( tup[0] - circle['center'][0] ) ** 2 +
  #             ( tup[1] - circle['center'][1] ) ** 2 -
  #                 circle['radius'] ** 2 ) < 2*s :
  #       used_xy.append(tup)
  #     else:
  #       allPoints.append(tup)
  #   data['allPoints'] = allPoints

  # data['allPoints'] += used_xy
  x,y = zip(*data['allPoints'])
  if VISUALISATION:
    plotData(x,y,circles,savePath='../img/2D_HT/result_%s_newsigma.pdf' % name)
    rcircles = getCirclesFromData(data)
    plotData(x,y,rcircles,savePath='../img/2D_HT/real_result_%s_newsigma.pdf' % name)



if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit( 'Please provide file to be read' )
  path = sys.argv[1]
  data = readFile(path)
  HoughTransform2D(data, path[8:-4])
