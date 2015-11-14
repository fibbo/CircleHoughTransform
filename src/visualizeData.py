import sys
import pprint
from Tools import readFile, S_OK, S_ERROR
import numpy as np
import matplotlib.pyplot as plt

def convertTuplesToList( tuples ):
  x,y = [],[]
  for atuple in tuples:
    x.append(atuple[0])
    y.append(atuple[1])

  return x,y

def plotData( x,y, circles, savePath=None ):
  fig = plt.gcf()
  for circle in circles:
    c = plt.Circle( circle['center'], circle['radius'], color='b',fill=False)
    print circle['center']
    plt.scatter( circle['center'][0], circle['center'][1], color='g', marker='v')
    fig.gca().add_artist(c)
  plt.scatter(x,y, color='r', marker='o')
  if savePath:
    fig.savefig(savePath)
  plt.show()

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'Please provide file to be read' )
  pp = pprint.PrettyPrinter(depth=6)
  path = sys.argv[1]
  data = readFile( path )
  x,y = convertTuplesToList(data['allPoints'])
  plotData( x,y )