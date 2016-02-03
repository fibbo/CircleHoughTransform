from math import sqrt
import matplotlib.pyplot as plt
import scipy.constants as sconst
import numpy as np
import pdb

from visualizeData import plotData
from Tools import *

DIMENSION=200
R_DIMENSION=100
THRESHOLD=3500

VISUALISATION=True

def HoughTransform3D( data, name ):
  x = np.linspace(-0.5,0.5,DIMENSION)
  y = np.linspace(-0.5,0.5,DIMENSION)
  r = np.linspace(0,0.5, R_DIMENSION)

  circles = []
  score = 0
  used_xy = []
  circle_counter = 0
  while True:
    weights = np.zeros( (R_DIMENSION, DIMENSION, DIMENSION))
    for x0,y0 in data['allPoints']:
      s = 0.001
      eta = (x[None,None,:]-x0)**2 + (y[None,:,None]-y0)**2 - r[:,None,None]**2
      weights += 1. / ( sqrt( 2 * sconst.pi ) * s ) * np.exp( -( eta ** 2 )\
                                                              / ( 2 * s ** 2 ) )
    index = np.argmax( weights )
    rr,jj,ii = np.unravel_index( index, (R_DIMENSION, DIMENSION, DIMENSION))
    score = weights.max()
    if score < THRESHOLD:
      print 'finished after %s circle(s) found' % circle_counter
      break
    print "(x,y,r): (%s,%s,%s)" % (ii, jj, rr)
    print "score: %s" % score
    circle_counter += 1
    circle = {}
    circle['center'] = (x[ii], y[jj])
    circle['radius'] = r[rr]
    circles.append(circle)

    used_xy += [tup for tup in data['allPoints'] if
                abs( ( tup[0] - circle['center'][0] ) ** 2 +
                     ( tup[1] - circle['center'][1] ) ** 2 -
                     circle['radius'] ** 2 ) < 2 * 0.001]
    data['allPoints'][:] = [tup for tup in data['allPoints'] if 
                            abs( ( tup[0] - circle['center'][0] ) ** 2 + 
                                 ( tup[1] - circle['center'][1] ) ** 2 - 
                                 circle['radius'] ** 2 ) >= 2 * 0.001]


    # plt.imshow(weights[rr][:][:])
    # plt.colorbar()
    # plt.show()

    # for r_i in range(R_DIMENSION):
    #     fig,ax1 = plt.subplots()
    #     img = ax1.imshow(weights[r_i][:][:], aspect='auto', interpolation='nearest')
    #     cba = plt.colorbar(img)
    #     plt.savefig('../img/3D_HT/slices/r_slices_%s_%s_circle%s.pdf' % (name, r_i, circle_counter))
    #     plt.close()
    
    # if VISUALISATION:
      # ux, uy = zip(*used_xy)
      # fig = plt.figure()
      # ax1 = fig.add_subplot(211)
      # ax1.scatter(ux, uy,c='r')
      # plt.xlim(-0.5,0.5)
      # plt.ylim(-0.5,0.5)
      # if data['allPoints']:
      #   ox, oy = zip(*data['allPoints'])
      #   ax2 = fig.add_subplot(212)
      #   ax2.scatter(ox, oy)
      # plt.xlim(-0.5,0.5)
      # plt.ylim(-0.5,0.5)
      # plt.show()
      # print ii,jj,rr

      # plt.show()
      
    
  data['allPoints'] += used_xy
  x,y = zip(*data['allPoints'])
  plotData(x,y,circles, savePath='../img/3D_HT/result_%s_%s.pdf' % (name, THRESHOLD))
  plotData(x,y,getCirclesFromData(data), savePath='../img/3D_HT/real_result_%s.pdf' % name)






if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit("Please provide file to be read")

  path = sys.argv[1]
  data = readFile(path)
  if 'Event' not in path:
    HoughTransform3D(data, path[8:-4])
  else:
    HoughTransform3D(data, path[-8:-4])
