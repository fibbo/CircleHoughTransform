# default python imports
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb


from visualizeData import plotData
from Tools import *

DIMENSION=1001

def HoughTransform1D( data ):
  
  r = np.linspace(0,1,DIMENSION)
  used_points = []
  res = []
  for center in data['Center']:
    weights = np.zeros(DIMENSION)
    for x0,y0 in data['allPoints']:
      s = 2*0.001
      eta = (center[0]-x0)**2 + (center[1]-y0)**2 - r**2
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
    plt.bar(range(DIMENSION),weights, width=2)
    plt.xlim(0,300)
    plt.show()
    index = np.argmax(weights)
    circle = {}
    circle['center'] = center
    circle['radius'] = r[index]
    res.append(circle)


  x,y = zip(*data['allPoints'])
  plotData(x,y,res)

def removePoints(data, center, r):
  used_points = []
  i = 0
  for x0, y0 in data['allPoints']:
    if abs( (x0-center[0])**2 + (y0-center[1])**2 - r**2 < 0.001):
      used_points.append( data['allPoints'].pop( i ) )
    else:
      i += 1
  return used_points

if __name__ == '__main__':

#### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'please provide file to be read' )
  path = sys.argv[1]
  data = readFile( path )

#### run specific methods ####
  # calculateWeights( data, gaussWeight, dim = 200 )
  # find2DCluster( data, neighbor_weights = False, weight_factor = 0.5 )

  # colors = ['red', 'blue', 'green', 'yellow', 'purple']
  # i = 0
  # for c1, c2 in data['CalcCenter']:
  #   print "(%s, %s) - %s" % ( c1, c2, colors[i] )
  #   i += 1
  # print data['GridSum']
  #visualize( data )
  HoughTransform1D(data)
  print 'The End'



  # h.printHistogramVariables()
