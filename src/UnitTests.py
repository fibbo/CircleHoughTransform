import unittest

from Tools import readFile
from CircleDetection3Points import *
import numpy as np
from Queue import Queue
import threading
import itertools
import matplotlib.pyplot as plt
from scipy import special


class ToolsTest( unittest.TestCase ):

  def setUp( self ):
    self.filename = '/home/phi/mt/HoughTransform/3_circles.txt'

    self.r = [ 0.3, 0.3, 0.40, 0.40, 0.50, 0.30, 0.3 ]
    self.c = [ (0.,0.), (0.,0.), (0.2,0.3), (0.1,0.5), (0.2,0.3), (0.7,0.), (0., 0.)]

    self.points = [ np.array((0.,0.3)), np.array((0.,-0.3)), np.array((0.3,0.)), np.array((-0.3,0.)) ]
    self.combinationsList =   list( itertools.combinations( self.points, 3 ) )

  def tstReadFile( self ):
    res = readFile( self.filename )
    print res

  def tstHistogram( self ):
    data = zip(self.r, self.c)
    dtype = [('radius', float), ('center', tuple)]
    data = np.array(data,dtype=dtype)
    res = SHistogram(data, 50, [0,0.5])
    H, bins, center = res['Value']
    print bins
    radius = {}
    radius['H'] = H
    print H
    radius['xedges'] = bins
    radius['center_data'] = center
    radiuses, centers = extractRadius( radius )

    for radius, center in zip(radiuses, centers):
      print radius, center

  def tstcalcPoints( self ):
    res = calculateCircleFromPoints( self.combinationsList )
    print res


  def tstCombListST( self ):
    data = readFile( '../data/lhcb_data/Event00009999.txt' )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r = calculateCircleFromPoints( combinationsList )
    print len(r)

  def tstBackgroundHistogram( self ):
    data = readFile( '../data/0_circles_200_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r1 = calculateCircleFromPoints( combinationsList, onlyRadius=True )


    data = readFile( '../data/0_circles_130_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r2 = calculateCircleFromPoints( combinationsList, onlyRadius=True )

    data = readFile( '../data/0_circles_150_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r3 = calculateCircleFromPoints( combinationsList, onlyRadius=True )

    data = readFile( '../data/0_circles_185_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r4 = calculateCircleFromPoints( combinationsList, onlyRadius=True )
  
    plt.hist(r1, 200, alpha=0.5, label='200')
    plt.hist(r2, 200, alpha=0.5, label='130')
    plt.hist(r3, 200, alpha=0.5, label='150')
    plt.hist(r4, 200, alpha=0.5, label='185')
    plt.legend(loc='upper right')
    plt.show()

  def tstScaling(self):
    factor = 1.72973
    data = readFile( '../data/0_circles_500_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r2 = calculateCircleFromPoints( combinationsList, onlyRadius=True )
    h2, edges = np.histogram( r2, 100, [0,1])
    np.savetxt('r2.txt', h2)
    h2 = factor*h2

    data = readFile( '../data/0_circles_600_bg.txt')
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r1 = calculateCircleFromPoints( combinationsList, onlyRadius=True )
    h1, edges = np.histogram( r1, 100, [0,1])

    # plt.hist(r1, 200, alpha=0.5, label='200')
    # plt.hist(r2, 200, alpha=0.5, label='130')
    # plt.legend(loc='upper right')
    # plt.show()
    step = 1./100
    edges = np.arange(0,1,step)
    plt.bar(edges,h1, width=step, alpha = 0.5, label='600')
    plt.bar(edges,h2, width=step, alpha = 0.5, label='500', color='y')
    plt.legend(loc='upper right')
    plt.xlim(min(edges), max(edges))
    
    plt.show()

  def testDiffBackground(self):
    background_hits = range(51)
    for i in background_hits:
      path = '../data/1_ring_diff_bg/1_circle_%s_bg.txt' % i
      data = readFile( path )
      combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
      print "%s bg hits" % i
      res = main( combinationsList )

  def tstDiffCircleHits( self ):
    circle_hits = range(5,30,5)
    for i in circle_hits:
      path = '../data/1_ring_diff_hits_const_bg/1_circle_%s_hits_20_bg.txt' % i
      data = readFile( path )
      combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
      print "%s circle hits" % i
      res = main( combinationsList )

if __name__ == '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromTestCase( ToolsTest )
  unittest.TextTestRunner( verbosity = 2 ).run( suite )
