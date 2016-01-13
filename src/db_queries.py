import pickle
import os
import numpy as np

import matplotlib.pyplot as plt

f = open('db.pkl')

db = pickle.load(f)

def findMaxRadius( db ):
  max_radius = 0
  filename = ''
  for entry in db:
    for ring in entry['rings']:
      if ring['radius'] > max_radius:
        max_radius = ring['radius']
        filename = entry['filename']
  return max_radius, filename

def findCircleWithLeastPoints( db ):
  nPe = 99
  for entry in db:
    for ring in entry['rings']:
      if ring['nPe'] < nPe:
        nPe = ring['nPe']
        filename = entry['filename']

  return nPe, filename

def findDataWithXCircles( db, circles ):
  results = []

  for entry in db:
    if len(entry['rings']) == circles:
      results.append( os.path.basename(entry['filename']) )


  return results

def getCircleData( db, eventnumber ):
  for ring in db[eventnumber]['rings']:
    print "------"
    print "center: %s" % (ring['center'],)
    print "nPe: %s" % ring['nPe']
    print "radius: %s" % ring['radius']

    

def pointsPerCircleDistribution( ):
  points = []
  for entry in db:
    for rings in entry['rings']:
      points.append(rings['nPe'])

  bins = np.linspace(1,45,45)
  plt.hist(points,bins)
  plt.show()

def ratioOfCirclesMoreThanXPoints(x=8):
  s_circle = 0
  b_circle = 0
  for entry in db:
    for ring in entry['rings']:
      if ring['nPe'] > x:
        b_circle += 1
      else:
        s_circle += 1

  return b_circle/float(s_circle+b_circle)


def plotRatioOfCirclesWithMoreThanXPoints():
  result = []
  for x in range(8,40):
    result.append(ratioOfCirclesMoreThanXPoints(x))

  plt.plot(result)
  plt.yticks(np.arange(0, 1, 0.1))
  plt.ylabel('Ratio of circles bigger than $x$ over the total number of circles')
  plt.show()


def ringsPerEventDistribution( db ):
  rings = []
  for entry in db:
    rings.append( len(entry['rings'] ) )

  print len(rings)
  bins = np.linspace(1,40,40)
  plt.hist(rings,bins)
  plt.show()

def maxPoints( db ):
  maxPoints = 0
  for i,event in enumerate(db):
    nPoints = 0
    nPoints += event['nBkg']
    for ring in event['rings']:
      nPoints += ring['nPe']

    if nPoints > maxPoints:
      maxPoints = nPoints
      eventnumber = i
  print eventnumber

# maxPoints( db )
#print db[3]
#ringsPerEventDistribution( db )
#pointsPerCircleDistribution(db)
#print findDataWithXCircles(db, 4)
#getCircleData( db, 9999)
# print findCircleWithLeastPoints(db)
#print findMaxRadius( db )
print db
# plotRatioOfCirclesWithMoreThanXPoints()
