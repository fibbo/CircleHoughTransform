import unittest

from Tools import readFile
from CircleDetection3Points import SHistogram, extractRadius, calculateCircleFromPoints, mt_calculateCircleFromPoints
import numpy as np
from Queue import Queue
import threading
import itertools

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

  def xtestCombListMT( self ):
    data = readFile( '../data/lhcb_data/Event00002346.txt' )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    q = Queue()
    for points in combinationsList:
      t = threading.Thread(target=mt_calculateCircleFromPoints, args = (q,points))
      t.daemon = True
      t.start()

  def testCombListST( self ):
    data = readFile( '../data/lhcb_data/Event00001486.txt' )
    combinationsList =   list( itertools.combinations( data['allPoints'], 3 ) )
    xy, r = calculateCircleFromPoints( combinationsList )

if __name__ == '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromTestCase( ToolsTest )
  unittest.TextTestRunner( verbosity = 2 ).run( suite )
