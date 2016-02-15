# default python imports
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb


from visualizeData import plotData
from Tools import *

DIMENSION=1001

def HoughTransform1D( data,name ):
  
  r = np.linspace(0,1,DIMENSION)
  used_points = []
  res = []
  counter = 1
  for center in data['Center']:
    weights = np.zeros(DIMENSION)
    for x0,y0 in data['allPoints']:
      s = 0.001
      eta = (center[0]-x0)**2 + (center[1]-y0)**2 - r**2
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 ) / ( 2 * s ** 2 ) )
    lines = plt.plot(np.linspace(0,1,DIMENSION),weights)
    plt.setp(lines, linewidth=1.5, color='b')
    plt.xlim(0.,0.5)
    # plt.ylim(0,4500)
    plt.xlabel('radius')
    plt.ylabel('score')
    plt.xticks(np.arange(0, 1, 0.1))


    # labels = np.linspace(0,1,21)
    # ax.set_xticklabels(labels)
    # plt.xlim(0,300)
    plt.savefig('../img/1D_HT/radius_scores_%s_%s.pdf' % (name, counter))
    plt.close()

    plt.bar(r,weights, width=0.001)
    plt.xlabel('radius')
    plt.ylabel('score')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.savefig('../img/1D_HT/radius_bar_scores_%s_%s.pdf' % (name, counter))
    plt.close()
    index = np.argmax(weights)
    clearRange = 20
    print "Score: %s" % weights[index]
    for i in range(index-clearRange,index+clearRange):
      try:
        weights[i] = 0
      except IndexError:
        continue

    index2 = np.argmax(weights)
    print "Second highest score: %s" % weights[index2]
    circle = {}
    circle['center'] = center
    circle['radius'] = r[index]
    res.append(circle)
    counter += 1

  x,y = zip(*data['allPoints'])
  plotData(x,y,res,savePath='../img/1D_HT/result_%s.pdf' % name)
  plotData(x,y,getCirclesFromData(data), savePath='../img/1D_HT/real_result_%s.pdf' % name)

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
  HoughTransform1D(data,path[8:-4])
  print 'The End'



  # h.printHistogramVariables()
