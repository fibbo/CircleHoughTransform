import pickle
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = open('../db.pkl')

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

    

def pointsPerCircleDistribution( db ):
  points = []
  for entry in db:
    for rings in entry['rings']:
      points.append(rings['nPe'])

  plt.hist(points, 30)
  plt.show()

def ringsPerEventDistribution( db ):
  rings = []
  for entry in db:
    rings.append( len(entry['rings'] ) )

  print len(rings)
  plt.hist(rings,15)
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

maxPoints( db )
#print db[3]
#ringsPerEventDistribution( db )
#pointsPerCircleDistribution(db)
#print findDataWithXCircles(db, 4)
getCircleData( db, 9999)
# print findCircleWithLeastPoints(db)
#print findMaxRadius( db )
