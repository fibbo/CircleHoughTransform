#default python imports
import sys
import math
import matplotlib.pyplot as plt

from Histogram import Histogram
from Tools import *

import pdb

def createHistograms(data):
  """ Fill a histogram for different radiuses with all the points for each circle center 
      :param double x, list of x coordinates
      :param double y, list of y coordinates
      :param list circle_centers, list of circle centers
      :returns a list of histograms. one for each circle center
  """

  histograms = []
  for a, b in data['Center']:
    h = Histogram(0.02, 0.29, 1.1, (a,b))
    radiuses = calcRadius( data, a, b )
    for radius in radiuses:
      h.addValue(radius)
    histograms.append(h)
  return histograms


def findRadiusClusters(histograms):
  """ search through the histograms for a cluster. when a cluster is found we save the radius value
      of that cluster in a list and return it.
  :param Histogram histograms: list of histograms
  :returns float radius: list of radiuses found
  

  """

  radiuses = []
  for h in histograms:
    clusters = []
    temp = []
    j = 0
    for entry in h.hist:
      if entry > 5:
        clusters.append(j)
      j += 1
    for cid in clusters:
      temp.append(h.getValue(cid))
    result = {}
    result['Radius'] = temp
    result['Center'] = h.center
    radiuses.append(result)

  return radiuses
        

def HoughTransform(data):

  histograms = createHistograms(data)
  radiuses = findRadiusClusters(histograms)
  plt.scatter(data['x'], data['y'])

  fig = plt.gcf()
  for radius in radiuses:
    #pdb.set_trace()
    if not radius['Radius']:
      continue
    
    for i in radius['Radius']:
      circle = plt.Circle(radius['Center'],i, fill=False)    
      fig.gca().add_artist(circle)

  #fig.gca().add_artist(plt.Circle((-0.342,-0.994),0.719,fill=False,color='red'))
  fig.gca().add_artist(plt.Circle((-0.821,-0.656),0.563,fill=False,color='green'))
  #fig.gca().add_artist(plt.Circle((-0.261,-0.328),0.892,fill=False,color='blue'))
  plt.show()


def calcRadius(data, a, b ):
    r = []
    x = data['x']
    y = data['y']
    for i in range(len(x)):
        radius = math.sqrt((x[i]-a)**2 + (y[i]-b)**2)
        r.append(radius)

    return r

if __name__ == '__main__':
  if len(sys.argv) < 2:
      sys.exit( 'please provide file to be read')
  path = sys.argv[1]
  data = readFile(path)
  centers = [(-0.342,-0.994), (-0.821,-0.656), (-0.261,-0.328)]
  data['Center'] = centers
  HoughTransform(data)
  

  
  #h.printHistogramVariables()
