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
  colors = ['b','g','r','c','m','y']
  font = {'weight' : 'normal',
        'size'   : 18}
  plt.rc('font', **font)
  plt.rcParams['lines.linewidth'] = 1.2
  my_dpi = 96
  plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
  plt.xlim(-0.5, 0.5)
  plt.ylim(-0.5, 0.5)

  
  fig = plt.gcf()
  plt.xlabel('$x$ [m]')
  plt.ylabel('$y$ [m]')
  i = 0
  for circle in circles:
    c = plt.Circle( circle['center'], circle['radius'], color=colors[i%6],fill=False)
    i+=1
    fig.gca().add_artist(c)
  plt.scatter(x,y, color='k', marker='o')
  plt.show()
  if savePath:
    fig.savefig(savePath)
  

if __name__ == '__main__': 
  #### read data #####
  if len( sys.argv ) < 2:
    sys.exit( 'Please provide file to be read' )
  pp = pprint.PrettyPrinter(depth=6)
  path = sys.argv[1]
  data = readFile( path )
  x,y = convertTuplesToList(data['allPoints'])
  plotData( x,y )