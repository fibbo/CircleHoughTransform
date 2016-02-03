import sys
import numpy as np
from ast import literal_eval

def norm( array1, array2 ):
  return np.sqrt( (array1[0]-array2[0])**2 + (array1[1]-array2[1])**2)

def readFile2( filename ):
  """ Used to open files generated by the HoughTransform code. It reads the x, y coordinates of
      the circle points into a list of list (points) and seperately into lists x and y

  :param str filename: file to be opened
  :returns dictionary with following keys: Points, x and y. Points contains

  """
  try:
    f = open( filename, 'r' )
  except Exception, e:
    sys.exit( e )
  center = []
  radius = []
  nPoints = []
  x = []
  y = []
  for line in f:
    if not line.startswith( '#' ):
        sep = line.split()
        x.append( float( sep[1] ) )
        y.append( float( sep[2] ) )
    if 'Center' in line:
      sep = line.split()
      tmp = literal_eval( sep[8] )
      center.append( tmp )

    if 'Radius' in line:
      sep = line.split()
      radius.append( float( sep[6] ) )
    if 'Center' in line:
      sep = line.split()
      nPoints.append( float( sep[3] ) )

  return { 'x' : x, 'y' : y, 'Center' : center, 'Radius' : radius, 'nPoints' : nPoints }


def getCirclesFromData(data):
  circles = []
  for c,r in zip(data['Center'],data['Radius']):
    circle = {}
    circle['center'] = c
    circle['radius'] = r
    circles.append(circle)
  return circles

def readFile( filename ):
  """ Reads the x, y coordinates of
      the circle points into a list of list (points) and seperately into lists x and y

  :param str filename: file to be opened
  :returns dictionary with following keys: Points, x and y. Points contains

  """
  try:
    f = open( filename, 'r' )
  except Exception, e:
    sys.exit( e )
  center = []
  radius = []
  nPoints = []
  allPoints = []
  for line in f:
    if not line.startswith( '#' ):
        sep = line.split()
        allPoints.append( np.array( [float( sep[1] ), float( sep[2] ) ] ) )
    if 'Radius' in line:
      sep = line.split()
      radius.append( float( sep[6] ) )
    if 'Center' in line:
      sep = line.split()
      center.append( np.array( (float( sep[9] ), float( sep[11] )) ) )

  return { 'allPoints' : allPoints, 'Center' : center, 'Radius' : radius, 'nPoints' : nPoints }


def S_ERROR( messageString = '' ):
  """ return value on error confition
  :param string messageString: error description
  """
  return { 'OK' : False, 'Message' : str( messageString )  }

def S_OK( value = None ):
  return { 'OK' : True, 'Value' : value }

def combinations(iterable, r):
  # combinations('ABCD', 2) --> AB AC AD BC BD CD
  # combinations(range(4), 3) --> 012 013 023 123
  pool = tuple(iterable)
  n = len(pool)
  if r > n:
      return
  indices = range(r)
  yield tuple(pool[i] for i in indices)
  while True:
    for i in reversed(range(r)):
      if indices[i] != i + n - r:
        break
    else:
      return
    indices[i] += 1
    for j in range(i+1, r):
        indices[j] = indices[j-1] + 1
    res = tuple(pool[i] for i in indices)
    max_distance = 0.15
    if np.linalg.norm(res[0]-res[1]) > max_distance or np.linalg.norm(res[0]-res[2]) > max_distance or np.linalg.norm(res[1]-res[2]) > max_distance:
      continue
    else:
      yield res

if __name__ == '__main__':
  a = [1,1]
  b = [3,5]
  print norm(a,b)