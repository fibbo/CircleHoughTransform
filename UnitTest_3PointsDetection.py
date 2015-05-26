import unittest
from src.CircleDetection3Points import findCircles, extractRadius, extractCenter
import numpy as np
class basicTest( unittest.TestCase ):
  def setUp(self):
    """ Setting up unittest """
    pass

  def testFindCircle(self):
    points = [(np.array([0,0.1]), np.array([0.1,0]), np.array([-0.1, 0]))]
    center, radius = findCircles(points)
    radius = extractRadius(radius)
    center = extractCenter(center)
    self.assertEqual( radius[0], 0.1, True)
    self.assertEqual( center[0], (0,0), True)






if __name__ == '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromTestCase( basicTest )
  unittest.TextTestRunner( verbosity = 2 ).run( suite )