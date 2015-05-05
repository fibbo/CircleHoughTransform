import unittest

from Tools import readFile


class ToolsTest( unittest.TestCase ):

  def setUp( self ):
    self.filename = '/home/phi/mt/HoughTransform/3_circles.txt'



  def testReadFile( self ):
    res = readFile( self.filename )
    print res


if __name__ == '__main__':
  suite = unittest.defaultTestLoader.loadTestsFromTestCase( ToolsTest )
  unittest.TextTestRunner( verbosity = 2 ).run( suite )
